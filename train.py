import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm

from dataset import create_wall_dataloader
from models import RecurrentJEPA

def check_for_collapse(embeddings: torch.Tensor, eps: float = 1e-8):
    if embeddings.dim() == 3:
        B,T,D = embeddings.shape
        flat_emb = embeddings.view(B*T, D)
    else:
        flat_emb = embeddings
    var = flat_emb.var(dim=0)
    mean_var = var.mean().item()
    print(f"Check collapse: avg var={mean_var:.6f}, min var={var.min().item():.6f}, max var={var.max().item():.6f}")
    if mean_var < eps:
        print("Warning: Potential collapse detected.")
    return mean_var

if __name__ == "__main__":
    data_path = "/scratch/DL24FA/train"  # Adjust
    batch_size = 64
    lr = 3e-4
    epochs = 20
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs("checkpoints", exist_ok=True)

    train_loader = create_wall_dataloader(
        data_path=data_path,
        probing=False,
        device=device,
        batch_size=batch_size,
        train=True,
    )

    model = RecurrentJEPA(state_dim=128, action_dim=2, proj_dim=128, hidden_dim=256, ema_rate=0.99, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            states = batch.states # [B,T,2,H,W]
            actions = batch.actions # [B,T-1,2]

            B,T,_,_,_ = states.shape

            # Unroll the model recurrently over time
            # We'll get embeddings for each timestep
            embeddings = []
            hidden_state = None
            for t in range(T):
                # For first step, no action. We can pass a dummy action of zeros if needed.
                # Actually, the recurrent code snippet suggests actions[:,t] even at t=0,
                # but actions is [B,T-1,2], so for t=0 we have no action. Let's define action_t:
                if t == 0:
                    # no previous action, just pass zeros
                    action_t = torch.zeros(B, 2, device=device)
                else:
                    action_t = actions[:, t-1] # t>0: use the previous action step

                emb = model(states=states[:, t], actions=action_t, hidden_state=hidden_state)
                embeddings.append(emb)
                hidden_state = emb

            pred_encs = torch.stack(embeddings, dim=1) # [B,T,D], D=state_dim

            # Compute BYOL loss:
            # Get target projections for states
            with torch.no_grad():
                target_proj = model.encode_target(states)    # [B,T,proj_dim]

            # Get online projections (and then predict)
            enc, online_proj = model.encode_online(states)   # [B,T,D], [B,T,proj_dim]

            # Predict target projection from online_proj
            # Flatten for predictor
            B,T,_ = online_proj.shape
            online_proj_flat = online_proj.reshape(B*T, -1)
            online_pred_flat = model.online_predictor(online_proj_flat)
            online_pred = online_pred_flat.view(B,T,-1) # [B,T,proj_dim]

            loss = model.compute_byol_loss(online_pred, target_proj)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.update_target_network()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        # Check for collapse once per epoch
        with torch.no_grad():
            mean_var = check_for_collapse(pred_encs)

        # Save model every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_path = f"checkpoints/epoch_{epoch+1}_jepa.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Model checkpoint saved at {save_path}")
