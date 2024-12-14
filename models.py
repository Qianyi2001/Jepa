from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch
from copy import deepcopy

class Encoder(nn.Module):
    def __init__(self, in_channels=2, state_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1),  # 65x65 -> 65x65
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),          # 65x65 -> 33x33
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),         # 33x33 -> 17x17
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),        # 17x17 -> 9x9
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, 2, 1),        # 9x9 -> 5x5
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 5 * 5, 1024),
            nn.ReLU(),
            nn.Linear(1024, state_dim)
        )

    def forward(self, x):
        # x: [B, C, H, W]
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        s = self.fc(h)
        return s

class TransitionModel(nn.Module):
    def __init__(self, state_dim=128, action_dim=2, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim * 2 + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, prev_embed, curr_embed, action):
        x = torch.cat([prev_embed, curr_embed, action], dim=-1)
        return self.net(x)

class Projection(nn.Module):
    def __init__(self, emb_dim=128, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, proj_dim)
        )

    def forward(self, x):
        return self.net(x)

class Predictor(nn.Module):
    def __init__(self, emb_dim=128, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim)
        )

    def forward(self, x):
        return self.net(x)

class RecurrentJEPA(nn.Module):
    def __init__(self, state_dim=128, action_dim=2, proj_dim=128, hidden_dim=512, ema_rate=0.99, device="cuda"):
        super().__init__()
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.proj_dim = proj_dim
        self.ema_rate = ema_rate

        # Online
        self.online_encoder = Encoder(in_channels=2, state_dim=state_dim)
        self.online_projection = Projection(emb_dim=state_dim, proj_dim=proj_dim)
        self.online_predictor = Predictor(emb_dim=proj_dim, hidden_dim=hidden_dim)

        # Target
        self.target_encoder = Encoder(in_channels=2, state_dim=state_dim)
        self.target_projection = Projection(emb_dim=state_dim, proj_dim=proj_dim)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_projection.parameters():
            p.requires_grad = False

        # Transition model
        self.transition_model = TransitionModel(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)

        self._update_target_network(initial=True)
        self.to(device)

    @torch.no_grad()
    def _update_target_network(self, initial=False):
        if initial:
            self.target_encoder.load_state_dict(self.online_encoder.state_dict())
            self.target_projection.load_state_dict(self.online_projection.state_dict())
        else:
            for online_params, target_params in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
                target_params.data = self.ema_rate * target_params.data + (1 - self.ema_rate) * online_params.data
            for online_params, target_params in zip(self.online_projection.parameters(), self.target_projection.parameters()):
                target_params.data = self.ema_rate * target_params.data + (1 - self.ema_rate) * online_params.data

    @torch.no_grad()
    def encode_target(self, states):
        B, T, C, H, W = states.shape
        flat = states.view(B * T, C, H, W)
        enc = self.target_encoder(flat)
        proj = self.target_projection(enc)
        proj = proj.view(B, T, -1)
        return proj

    def encode_online(self, states):
        B, T, C, H, W = states.shape
        flat = states.view(B * T, C, H, W)
        enc = self.online_encoder(flat)
        proj = self.online_projection(enc)
        enc = enc.view(B, T, -1)
        proj = proj.view(B, T, -1)
        return enc, proj

    def compute_byol_loss(self, online_pred, target_proj):
        B, T, D = online_pred.shape
        o_norm = F.normalize(online_pred.view(B * T, D), dim=-1)
        t_norm = F.normalize(target_proj.view(B * T, D).detach(), dim=-1)
        loss = 2 - 2 * (o_norm * t_norm).sum(dim=-1)
        return loss.mean()

    def update_target_network(self):
        self._update_target_network(initial=False)

    def forward(self, states, actions, hidden_state=None):
        e = self.online_encoder(states)
        if hidden_state is None:
            return e
        else:
            next_embed = self.transition_model(hidden_state, e, actions)
            return next_embed

class Prober(torch.nn.Module):
    def __init__(
        self,
        embedding: int,
        arch: str,
        output_shape: List[int],
    ):
        super().__init__()
        self.output_dim = np.prod(output_shape)
        self.output_shape = output_shape
        self.arch = arch

        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        f = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(f) - 2):
            layers.append(torch.nn.Linear(f[i], f[i + 1]))
            layers.append(torch.nn.ReLU(True))
        layers.append(torch.nn.Linear(f[-2], f[-1]))
        self.prober = torch.nn.Sequential(*layers)

    def forward(self, e):
        output = self.prober(e)
        return output
def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)

