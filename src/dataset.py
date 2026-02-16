import random
import time
import torch
from torch import Tensor
from torch.utils.data import Dataset
from src.modules.features import spatial_encoding, temporal_encoding
from src.synthetic_trajectory import SyntheticTrajectoryGenerator


class SyntheticTrajectoryDataset(Dataset):
    """
    Synthetic trajectory dataset for delta prediction.

    Each trajectory is generated using randomized starting coordinates, timestamps,
    and motion parameters. The dataset provides spatial and temporal feature tensors
    as inputs and delta (Δlat, Δlng) targets for supervised learning.

    :param num_trajectories: Number of synthetic trajectories to generate.
    :param center_lat: Reference latitude for the generation area.
    :param center_lng: Reference longitude for the generation area.
    """

    def __init__(self, num_trajectories: int, center_lat: float = 56.0, center_lng: float = 8.0):
        self.trajs: list[tuple[Tensor, Tensor]] = []
        self.labels: list[Tensor] = []
        self.original_trajs: list[list[tuple[float, float, int]]] = []

        for _ in range(num_trajectories):
            seed = random.randint(0, int(1e6))
            random.seed(seed)

            generator = SyntheticTrajectoryGenerator(
                start_lat=random.uniform(55.5, 56.5),
                start_lng=random.uniform(7.5, 8.5),
                start_timestamp=int(time.time()) + random.randint(-1000, 1000),
                step_seconds=60,
                num_points=random.randint(10, 49),
                delta_lat=random.uniform(-0.001, 0.001),
                enable_turns=True,
                delta_lng=random.uniform(-0.001, 0.001),
                noise=random.uniform(0.0, 0.0001),
                jitter_time=True,
                seed=seed,
            )

            traj = generator.generate()
            coords = torch.tensor([[p[0], p[1]] for p in traj], dtype=torch.float32)
            times = torch.tensor([p[2] for p in traj], dtype=torch.float64)
            temporal_tensor = temporal_encoding(times[:-1])
            spatial_deltas = (coords[1:] - coords[:-1]) * 1000
            temporal_deltas = (times[1:] - times[:-1]) / 60
            temporal_deltas = temporal_deltas.to(torch.float32)
            deltas = torch.cat([spatial_deltas, temporal_deltas.unsqueeze(-1)], dim=-1)
            spatial_tensor = spatial_encoding(coords[:-1])
            assert len(spatial_tensor) == len(deltas)

            self.trajs.append((spatial_tensor, temporal_tensor))
            self.labels.append(deltas)
            self.original_trajs.append(traj)

    def __len__(self) -> int:
        """
        :return: Number of trajectories in the dataset.
        """
        return len(self.trajs)

    def __getitem__(self, idx: int) -> tuple[list[list[tuple[float, float, int]]], tuple[Tensor, Tensor], Tensor]:
        """
        :param idx: Index of the trajectory sample to retrieve.
        :return: ((spatial_tensor, temporal_tensor), delta_tensor)
        """
        spatial, temporal = self.trajs[idx]
        deltas = self.labels[idx]
        return self.original_trajs, (spatial, temporal), deltas
