import torch
from matplotlib import pyplot as plt
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from new.encoder import trajectory_to_annotated_trajectory
from new.generator import SyntheticTrajectoryGenerator, SyntheticTrajectoryDataset


class TrajectoryEncoder(nn.Module):
    def __init__(self, input_dimension: int, embedding_dim: int):
        super().__init__()
        self.linear_layer = nn.Linear(input_dimension, embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear_layer(x)
def causal_mask(size: int) -> torch.Tensor:
    return torch.triu(torch.ones(size, size), diagonal=1).bool()




def collate_next_point_prediction(batch):
    trajs = [t for t, _ in batch]
    padded = pad_sequence(trajs, batch_first=False)  # [T, B, D]
    return padded

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.max_len = max_len

    def forward(self, x: Tensor) -> Tensor:
        # x: [T, B, D]
        T, B, D = x.shape
        if T > self.max_len:
            raise ValueError(f"Sequence length {T} exceeds max_len {self.max_len}")
        positions = torch.arange(T, device=x.device).unsqueeze(1).expand(T, B)  # [T, B]
        pos_embed = self.pos_embedding(positions)  # [T, B, D]
        return x + pos_embed

class TrajectoryPredictor(nn.Module):
    def __init__(self, input_dim, embed_dim, n_heads, num_layers, max_len=512, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.pos_encoder = LearnedPositionalEncoding(d_model=embed_dim, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(embed_dim, input_dim)

    def forward(self, x: Tensor) -> Tensor:
        # x: [T, B, D]
        T, B, D = x.shape
        x = self.input_proj(x)         # [T, B, E]
        x = self.pos_encoder(x)        # [T, B, E]
        mask = causal_mask(T).to(x.device)  # [T, T]

        x = self.transformer(x, mask=mask)  # [T, B, E]
        return self.output_proj(x)          # [T, B, D]

@torch.no_grad()
def sample_next_points(model, start_seq: Tensor, steps: int) -> Tensor:
    """
    start_seq: [T0, D] - initial annotated trajectory
    steps: number of future points to predict
    Returns: [T0 + steps, D]
    """
    model.eval()
    generated = [start_seq]  # list of [T0, D]
    current = start_seq.unsqueeze(1)  # [T0, 1, D]

    for _ in range(steps):
        pred = model(current)  # [T, 1, D]
        next_point = pred[-1, 0]  # last predicted point
        next_point = next_point.unsqueeze(0).unsqueeze(1)  # [1, 1, D]
        current = torch.cat([current, next_point], dim=0)
        generated.append(next_point.squeeze(1))  # [1, D]

    return torch.cat(generated, dim=0)  # [T0 + steps, D]
def plot_trajectory(predicted: Tensor, observed: Tensor):
    pred_xy = predicted[:, :2].cpu().numpy()
    obs_xy = observed[:, :2].cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.plot(obs_xy[:, 0], obs_xy[:, 1], 'bo-', label='Observed', alpha=0.7)
    plt.plot(pred_xy[:, 0], pred_xy[:, 1], 'ro--', label='Predicted', alpha=0.7)
    plt.scatter(obs_xy[0, 0], obs_xy[0, 1], c='black', marker='x', label='Start')
    plt.legend()
    plt.title("Trajectory Prediction")
    plt.xlabel("x (normalized UTM)")
    plt.ylabel("y (normalized UTM)")
    plt.axis("equal")
    plt.grid(True)
    plt.show()
def generate_and_plot(model):
    gen = SyntheticTrajectoryGenerator(delta_lat=0.01, delta_lng=0.0, seed=999)
    traj = gen.generate()
    center_lat, center_lng = 56.0, 8.0
    annotated = trajectory_to_annotated_trajectory(traj, center_lat, center_lng)

    prefix_len = 8
    input_prefix = annotated[:prefix_len]  # [T0, D]
    predicted = sample_next_points(model, input_prefix, steps=10)  # [T0+10, D]

    plot_trajectory(predicted, annotated[:prefix_len + 10])
def plot_dataset_samples(dataset, num_samples=5):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 6))
    for i in range(num_samples):
        traj_tensor, label = dataset[i]
        xy = traj_tensor[:, :2].numpy()  # normalized UTM (x, y)
        plt.plot(xy[:, 0], xy[:, 1], label=f"Traj {i} (Label={label})", marker='o', alpha=0.7)

    plt.title("Sample Trajectories from Dataset")
    plt.xlabel("Normalized UTM X")
    plt.ylabel("Normalized UTM Y")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.show()

def train_predictor() -> nn.Module:
    dataset = SyntheticTrajectoryDataset(num_trajectories=3000)

    plot_dataset_samples(dataset)

    dataloader = DataLoader(
        dataset, batch_size=16, shuffle=True, collate_fn=collate_next_point_prediction
    )

    sample_input, _ = dataset[0]
    input_dim = sample_input.shape[1]

    model = TrajectoryPredictor(
        input_dim=input_dim, embed_dim=64, n_heads=4, num_layers=2
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(100):
        model.train()
        total_loss = 0.0

        for x in dataloader:
            target = x[1:]       # [T-1, B, D]
            input_seq = x[:-1]   # [T-1, B, D]
            pred = model(input_seq)

            loss = loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    return model

if __name__ == "__main__":
    model = train_predictor()
    generate_and_plot(model)
    generate_and_plot(model)
    generate_and_plot(model)
