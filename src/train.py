import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from src.dataset import SyntheticTrajectoryDataset
from src.modules.model import CheminTF
from src.predict import predict_autoregressive
from src.plot import plot_trajectories


# ============================================================
# Collate function
# ============================================================

def collate_batch(batch: list[tuple[list, tuple[Tensor, Tensor], Tensor]]) -> tuple[list, Tensor, Tensor, Tensor]:
    """Pad variable-length trajectories and stack into [T, B, D] batches."""
    original, spatial_seqs, temporal_seqs, delta_seqs = zip(*[(t, s, t, d) for (t, (s, t), d) in batch])
    max_len = max(seq.shape[0] for seq in spatial_seqs)

    def pad_sequence(seq_list: list[Tensor]) -> Tensor:
        padded = [
            torch.cat([seq, torch.zeros(max_len - seq.shape[0], seq.shape[1])], dim=0)
            for seq in seq_list
        ]
        return torch.stack(padded, dim=1)  # [T, B, D]

    spatial_batch = pad_sequence(spatial_seqs)
    temporal_batch = pad_sequence(temporal_seqs)
    delta_batch = pad_sequence(delta_seqs)
    return original, spatial_batch, temporal_batch, delta_batch



# ============================================================
# Training function
# ============================================================
def train_model(
    num_epochs: int = 20,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> CheminTF:
    """
    Train CheminTF to predict spatial deltas (Δlat, Δlng, Δt) from spatio-temporal features.
    Adds a validation set and uses a validation trajectory for plotting.
    """
    # ---------------------- Dataset split ----------------------
    full_dataset = SyntheticTrajectoryDataset(num_trajectories=100000)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    # ---------------------- Model setup ----------------------
    model = CheminTF(n_heads=4, num_layers=2).to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # ---------------------- Metric storage ----------------------
    train_losses, val_losses, val_rmses = [], [], []

    *_, deltas_ref = val_dataset[0]


    # ---------------------- Training loop ----------------------
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for _, spatial, temporal, deltas in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            spatial, temporal, deltas = spatial.to(device), temporal.to(device), deltas.to(device)
            optimizer.zero_grad()
            preds = model(spatial, temporal)
            loss = criterion(preds, deltas)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ---------------------- Validation loop ----------------------
        model.eval()
        val_loss, val_rmse = 0.0, 0.0
        with torch.no_grad():
            for _, spatial, temporal, deltas in val_loader:
                spatial, temporal, deltas = spatial.to(device), temporal.to(device), deltas.to(device)
                preds = model(spatial, temporal)
                loss = criterion(preds, deltas)
                val_loss += loss.item()
                val_rmse += torch.sqrt(torch.mean((preds - deltas) ** 2)).item()
        avg_val_loss = val_loss / len(val_loader)
        avg_val_rmse = val_rmse / len(val_loader)
        val_losses.append(avg_val_loss)
        val_rmses.append(avg_val_rmse)

        print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.6f} | "
              f"Val Loss: {avg_val_loss:.6f} | Val RMSE: {avg_val_rmse:.4f}")

        # 🔁 Visualize one autoregressive prediction each epoch using VALIDATION reference
        with torch.no_grad():
            coord = val_dataset[0][0][0]  # original trajectory
            spatial_ref = Tensor(Tensor(coord)[:, :2])
            # temporal ref must be float64
            temporal_ref = Tensor(Tensor(coord)[:, 2]).to(torch.float64)

            pred_coords, pred_times = predict_autoregressive(
                model,
                init_coords=spatial_ref,
                init_times=temporal_ref,
                num_future_steps=10,
                device=device,
            )
            plot_trajectories(spatial_ref, pred_coords, epoch)

    # ---------------------- Plot training & validation curves ----------------------
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    plt.plot(val_losses, label="Val Loss", linewidth=2)
    plt.plot(val_rmses, label="Val RMSE", linewidth=2)
    plt.title("Training & Validation Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return model



if __name__ == "__main__":
    trained_model = train_model(num_epochs=100, batch_size=128, device="cuda")
    torch.save(trained_model.state_dict(), "../weights/chemin_tf_deltas.pt")
