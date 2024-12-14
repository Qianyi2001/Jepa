from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch
from copy import deepcopy

def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)

class Encoder(nn.Module):
    def __init__(self, in_channels=2, state_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1),
            nn.GELU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.GELU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.GELU(),
        )
        self.fc = nn.Linear(128 * 8 * 8, state_dim)

    def forward(self, x):
        # x: [B, C, H, W]
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        s = self.fc(h)
        return s

class TransitionModel(nn.Module):
    """
    Given the previous embedding, current state embedding, and action,
    predicts the next embedding. This is the recurrent step.
    """
    def __init__(self, state_dim=128, action_dim=2, hidden_dim=256):
        super().__init__()
        # Input: prev_embedding + current_state_embedding + action
        # Output: next embedding (state_dim)
        self.net = nn.Sequential(
            nn.Linear(state_dim*2 + action_dim, hidden_dim),
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
            nn.Linear(emb_dim, proj_dim)
        )

    def forward(self, x):
        # x: [N, emb_dim]
        return self.net(x)

class Predictor(nn.Module):
    def __init__(self, emb_dim=128, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim)
        )

    def forward(self, x):
        return self.net(x)

class RecurrentJEPA(nn.Module):
    """
    Recurrent JEPA model with BYOL-style components.
    Each forward call processes one timestep.
    """
    def __init__(self, state_dim=128, action_dim=2, proj_dim=128, hidden_dim=256, ema_rate=0.99, device="cuda"):
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

        # Transition model for recurrent update
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
        # states: [B,T,C,H,W]
        B,T,C,H,W = states.shape
        flat = states.view(B*T, C, H, W)
        enc = self.target_encoder(flat)           # [B*T, state_dim]
        proj = self.target_projection(enc)        # [B*T, proj_dim]
        proj = proj.view(B,T,-1)
        return proj

    def encode_online(self, states):
        # states: [B,T,C,H,W]
        B,T,C,H,W = states.shape
        flat = states.view(B*T, C, H, W)
        enc = self.online_encoder(flat)           # [B*T, state_dim]
        proj = self.online_projection(enc)        # [B*T, proj_dim]
        enc = enc.view(B,T,-1)
        proj = proj.view(B,T,-1)
        return enc, proj

    def compute_byol_loss(self, online_pred, target_proj):
        # online_pred, target_proj: [B,T,proj_dim]
        B,T,D = online_pred.shape
        # Normalize
        o_norm = F.normalize(online_pred.view(B*T,D), dim=-1)
        t_norm = F.normalize(target_proj.view(B*T,D).detach(), dim=-1)
        loss = 2 - 2 * (o_norm * t_norm).sum(dim=-1)
        return loss.mean()

    def update_target_network(self):
        self._update_target_network(initial=False)

    def forward(self, states, actions, hidden_state=None):
        """
        Forward one timestep:
        states: [B,C,H,W]
        actions: [B,2]
        hidden_state: [B, state_dim] or None

        Returns next embedding [B, state_dim].
        """
        # Encode current state
        e = self.online_encoder(states)  # [B, state_dim]

        if hidden_state is None:
            # First step, just use the current state embedding
            return e
        else:
            # Subsequent steps, use transition model
            next_embed = self.transition_model(hidden_state, e, actions)
            return next_embed

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        
        self.patch_embedding = nn.Conv2d(
            in_channels=in_channels, # first channel is agent, second is border and walls
            out_channels=embed_dim, # image size is 64 x 64, and we want
            kernel_size=patch_size, # 4x4 patches
            stride=patch_size # non-overlapping patches
        )
        
    def forward(self, x):
        x = self.patch_embedding(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)

        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()

        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.scale = self.head_dim ** -0.5
    
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attention_weights = torch.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)

        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, embed_dim)

        output = self.out_proj(attention_output)

        return output

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout):
        super(TransformerBlock, self).__init__()

        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        attention_output = self.attention(x)
        x = self.norm1(x + self.dropout(attention_output))

        mlp_output = self.mlp(x)
        x = self.norm2(x + self.dropout(mlp_output))

        return x

class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim, num_heads, mlp_dim, num_layers, num_classes, dropout=0.1):
        super(VisionTransformer, self).__init__()

        self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)

        num_patches = (image_size // patch_size) ** 2

        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = x + self.pos_embedding
        x = self.dropout(x)

        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        return x

class JEPAEncoder(torch.nn.Module):

    def __init__(self, device="cuda", bs=64, n_steps=17, output_dim=256):
        super().__init__()
        self.device = device
        self.bs = bs # batch size
        self.n_steps = n_steps # number of forward passes
        self.repr_dim = output_dim # encoder output dimension

        # need to make sense later
        self.encoder = VisionTransformer(
            image_size=64,
            patch_size=4,
            in_channels=2,
            embed_dim=256,
            num_heads=8,
            mlp_dim=512,
            num_layers=6,
            num_classes=256
        )

    def forward(self, states, actions):
        """
        Args:
            states: [B, T, Ch, H, W]
            actions: [B, T-1, 2]

        Output:
            predictions: [B, T, D]
        """
        B, T, C, H, W = states.size()
        states = states.view(B * T, C, H, W)  

        embeddings = self.encoder(states)  
        embeddings = embeddings.view(B, T, -1)  
        
        return embeddings  

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
