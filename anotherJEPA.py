import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import time

#########################
# Dataset and Dataloader
#########################

class TrajectoryDataset(Dataset):
    def __init__(self, states_path, actions_path):
        self.states = np.load(states_path)  # shape (N, T, 2, 64, 64)
        self.actions = np.load(actions_path) # shape (N, T-1, 2)
        
        self.states = torch.tensor(self.states, dtype=torch.float32)
        self.actions = torch.tensor(self.actions, dtype=torch.float32)
        
    def __len__(self):
        return self.states.shape[0]
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

#########################
# Model Components
#########################

class Encoder(nn.Module):
    def __init__(self, in_channels=2, state_dim=128):
        super().__init__()
        # Simple CNN encoder
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
        )
        # After downsampling 64x64 → approximately 8x8 feature map
        self.fc = nn.Linear(128*8*8, state_dim)
        
    def forward(self, x):
        # x: (B, T, C, H, W) or (B, C, H, W)
        if x.ndimension() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B*T, C, H, W)
            h = self.conv(x)
            h = h.view(h.size(0), -1)
            s = self.fc(h)
            s = s.view(B, T, -1)
        else:
            h = self.conv(x)
            h = h.view(h.size(0), -1)
            s = self.fc(h)
        return s

class Predictor(nn.Module):
    def __init__(self, state_dim=128, action_dim=2, hidden_dim=128):
        super().__init__()
        # Predictor takes s_{n-1} and u_{n-1}, outputs predicted s_n
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
    def forward(self, prev_state, prev_action):
        x = torch.cat([prev_state, prev_action], dim=-1)
        return self.fc(x)

#########################
# JEPA Model
#########################

class JEPA(nn.Module):
    def __init__(self, state_dim=128, action_dim=2, hidden_dim=128, ema_rate=0.99):
        super().__init__()
        # Online encoder (learned)
        self.online_encoder = Encoder(in_channels=2, state_dim=state_dim)
        
        # This code uses a BYOL-like EMA target encoder to stabilize training.
        # Target encoder (EMA copy of online encoder)
        self.target_encoder = deepcopy(self.online_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        
        # Predictor
        self.predictor = Predictor(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)
        
        self.ema_rate = ema_rate
        
    @torch.no_grad()
    def update_target_encoder(self):
        for online_params, target_params in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_params.data = self.ema_rate * target_params.data + (1 - self.ema_rate) * online_params.data
    
    def forward(self, states, actions):
        # states: (B, T, 2, 64, 64)
        # actions: (B, T-1, 2)
        
        # Encode states with online encoder
        online_states = self.online_encoder(states)  # (B, T, D_s)
        
        # Encode states with target encoder (no grad)
        with torch.no_grad():
            target_states = self.target_encoder(states)  # (B, T, D_s)
        
        # Predict future states in embedding space
        predicted_states_list = []
        T = online_states.shape[1]
        for t in range(1, T):
            prev_state = online_states[:, t-1, :]   # s_{n-1}
            prev_action = actions[:, t-1, :]        # u_{n-1}
            predicted_state = self.predictor(prev_state, prev_action)
            predicted_states_list.append(predicted_state)
        
        predicted_states = torch.stack(predicted_states_list, dim=1)  # (B, T-1, D_s)
        target_next_states = target_states[:, 1:, :]                  # (B, T-1, D_s)
        
        return predicted_states, target_next_states


#########################
# Training Loop Example
#########################

if __name__ == "__main__":
    device = (
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )

    # Hyperparams
    batch_size = 8
    lr = 3e-4
    epochs = 5
    state_dim = 128
    action_dim = 2
    hidden_dim = 128
    
    # Load data
    train_dataset = TrajectoryDataset("subset_states.npy", "subset_actions.npy")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    model = JEPA(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim, ema_rate=0.99).to(device)
    if device == 'cuda':
        model = torch.compile(model)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for states, actions in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            t0 = time.time()
            states = states.to(device)
            actions = actions.to(device)
            
            optimizer.zero_grad()
            predicted_states, target_states = model(states, actions)
            
            # Compute loss: distance between predicted and target embeddings
            with torch.autocast(device_type=device, dtype=torch.float16):
                loss = criterion(predicted_states, target_states)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Update target encoder
            with torch.no_grad():
                model.update_target_encoder()

            if device == 'mps':
                torch.mps.synchronize()
            elif device == 'cuda':
                torch.cuda.synchronize()

            t1 = time.time()
            dt = (t1 - t0) * 1000
            
            total_loss += loss.item()
            print(f"loss {loss.item()}, dt {dt:.2f}ms, norm {norm:.4f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
