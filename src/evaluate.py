import torch
from torch.utils.data import DataLoader

from src.dataset import SyntheticTrajectoryDataset
from src.modules.model import CheminTF
from src.train import collate_batch, predict_autoregressive, plot_trajectories


def evaluate_and_plot(
    weights_path: str = "chemin_tf_deltas.pt",
    num_samples: int = 10,
    num_trajectories: int = 1000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Load pretrained CheminTF model and plot autoregressive trajectory predictions.

    Args:
        weights_path: Path to model checkpoint (.pt file)
        num_samples: Number of trajectories to predict and plot
        num_trajectories: Number of synthetic trajectories in dataset
        device: "cuda" or "cpu"
    """
    # ============================================================
    # Load model and weights
    # ============================================================
    model = CheminTF(n_heads=4, num_layers=2).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # ============================================================
    # Prepare validation data
    # ============================================================
    dataset = SyntheticTrajectoryDataset(num_trajectories=num_trajectories)
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_batch)
    val_iter = iter(val_loader)

    # ============================================================
    # Predict and plot samples
    # ============================================================
    for i in range(num_samples):
        original, spatial, temporal, delta = next(val_iter)
        spatial_ref = spatial.squeeze(1).to(device)
        temporal_ref = temporal.squeeze(1).to(device)
        coords_ref = spatial_ref[:, :2]
        times_ref = temporal_ref[:, 0] if temporal_ref.ndim > 1 else temporal_ref
        times_ref = times_ref.to(torch.float64)
        pred_coords, pred_times = predict_autoregressive(
            model,
            init_coords=coords_ref,
            init_times=times_ref,
            num_future_steps=5,
            device=device,
        )

        plot_trajectories(coords_ref, pred_coords, i)
        print(f"Trajectory {i+1}/{num_samples} plotted ✅")

    print("✅ Evaluation and plotting complete.")


# ============================================================
# Example usage
# ============================================================
if __name__ == "__main__":
    evaluate_and_plot("../weights/chemin_tf_deltas.pt", num_samples=10)
