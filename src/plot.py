import torch
import matplotlib.pyplot as plt

def plot_trajectories(true_coords: torch.Tensor, pred_coords: torch.Tensor, epoch: int):
        true_coords = true_coords.cpu().numpy()
        pred_coords = pred_coords.cpu().numpy()

        plt.figure(figsize=(5, 5))
        plt.plot(true_coords[:, 0], true_coords[:, 1], "b.-", label="True")
        plt.plot(pred_coords[len(true_coords)-1::, 0, ], pred_coords[len(true_coords)-1:, 1, ], "orange", label="Predicted")
        plt.scatter(true_coords[0, 0], true_coords[0, 1], c="green", label="Start", zorder=5)
        plt.scatter(pred_coords[-1, 0], pred_coords[-1, 1], c="red", label="Pred End", zorder=5)
        plt.title(f"Autoregressive Prediction — Epoch {epoch+1}")
        plt.xlabel("Latitude")
        plt.ylabel("Longitude")
        plt.legend()
        plt.axis("equal")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
