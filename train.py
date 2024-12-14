import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm

from dataset import create_wall_dataloader
from models import RecurrentJEPA

def check_for_collapse(embeddings: torch.Tensor, eps: float = 1e-8):
    if embeddings.dim() == 3:
        B, T, D = embeddings.shape
        flat_emb = embeddings.view(B * T, D)
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
    lr = 1e-4
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

    model = RecurrentJEPA(state_dim=128, action_dim=2, proj_dim=128, hidden_dim=512, ema_rate=0.99, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Initialize scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=3,  # 第一次余弦周期的步长
        T_mult=2,  # 每次重启后周期倍增的系数
        eta_min=1e-6,  # 最小学习率
        last_epoch=-1,  # 恢复训练时的上次 epoch
        verbose=True  # 打印调度日志
    )

    batch_counter = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            states = batch.states  # [B,T,2,H,W]
            actions = batch.actions  # [B,T-1,2]

            B, T, _, _, _ = states.shape

            # Unroll the model recurrently over time
            embeddings = []
            hidden_state = None
            for t in range(T):
                if t == 0:
                    action_t = torch.zeros(B, 2, device=device)
                else:
                    action_t = actions[:, t-1]

                emb = model(states=states[:, t], actions=action_t, hidden_state=hidden_state)
                embeddings.append(emb)
                hidden_state = emb

            pred_encs = torch.stack(embeddings, dim=1)  # [B,T,D], D=state_dim

            # Compute BYOL loss:
            with torch.no_grad():
                target_proj = model.encode_target(states)  # [B,T,proj_dim]

            enc, online_proj = model.encode_online(states)  # [B,T,D], [B,T,proj_dim]

            B, T, _ = online_proj.shape
            online_proj_flat = online_proj.reshape(B * T, -1)
            online_pred_flat = model.online_predictor(online_proj_flat)
            online_pred = online_pred_flat.view(B, T, -1)  # [B,T,proj_dim]

            loss = model.compute_byol_loss(online_pred, target_proj)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Update scheduler
            scheduler.step()

            model.update_target_network()

            total_loss += loss.item()
            batch_counter += 1

            if batch_counter % 30 == 0:
                with torch.no_grad():
                    mean_var = check_for_collapse(pred_encs)
                print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

        save_path = f"checkpoints/epoch_{epoch}_jepa.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Model checkpoint saved at {save_path}")
        print(f"-----------Epoch {epoch + 1}, Loss: {total_loss:.4f}-------------")
