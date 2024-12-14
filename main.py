from dataset import create_wall_dataloader
from evaluator import ProbingEvaluator
import torch
from models import RecurrentJEPA
import glob


def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


def load_data(device):
    data_path = "DL24FA"

    probe_train_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/train",
        probing=True,
        device=device,
        train=True,
    )

    probe_val_normal_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_wall_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_wall/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_ds = {"normal": probe_val_normal_ds, "wall": probe_val_wall_ds}

    return probe_train_ds, probe_val_ds


def load_model(model_path):
    """Load or initialize the JEPA model and weights."""
    # Use the same parameters as during training
    model = RecurrentJEPA(state_dim=128, action_dim=2, hidden_dim=128, ema_rate=0.99)
    state_dict = torch.load("checkpoints/epoch_10_jepa.pth", map_location="cuda")
    model.load_state_dict(state_dict)
    model.to("cuda")
    model.eval()
    return model


def evaluate_model(device, model, probe_train_ds, probe_val_ds):
    evaluator = ProbingEvaluator(
        device=device,
        model=model,
        probe_train_ds=probe_train_ds,
        probe_val_ds=probe_val_ds,
        quick_debug=False,
    )

    prober = evaluator.train_pred_prober()

    avg_losses = evaluator.evaluate_all(prober=prober)

    for probe_attr, loss in avg_losses.items():
        print(f"{probe_attr} loss: {loss}")


if __name__ == "__main__":
    checkpoints = "checkpoints"
    device = get_device()
    for cp in glob.glob(f"{checkpoints}/*.pth"):
        model = load_model(cp)
        probe_train_ds, probe_val_ds = load_data(device)
        evaluate_model(device, model, probe_train_ds, probe_val_ds)
    # probe_train_ds, probe_val_ds = load_data(device)
    # model = load_model()
    # evaluate_model(device, model, probe_train_ds, probe_val_ds)
