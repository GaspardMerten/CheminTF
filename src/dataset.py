import os
from functools import lru_cache
from typing import List

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset

from src import config

FlattenedTrajectory = List[float]


def trajectory_as_patches(
    trajectory: FlattenedTrajectory, patch_size: int
) -> np.ndarray:
    """
    Convert a flattened trajectory into a list of patches. A flatten trajectory is a list of floats
    where the successive elements are the coordinates and the timestamp of the trajectory. For instance:
    [lat1, lon1, timestamp1, lat2, lon2, timestamp2, ...]
    or more generally:
    [lat1, lon1, *attribute1, timestamp1, lat2, lon2, *attribute2, timestamp2, ...]
    """
    patches = []
    actual_patch_size = patch_size * config.VALUES_PER_POINT

    for patch_start in range(0, len(trajectory) - actual_patch_size + 1, actual_patch_size):
        patch_end = patch_start + actual_patch_size
        patch = trajectory[patch_start:patch_end]
        if len(patch) < actual_patch_size:
            padding = config.NEUTRAL_VECTOR * (actual_patch_size - len(patch))
            patch = list(patch) + padding
        patches.append(patch)
    return np.array(patches, dtype=np.float32)


class TrajectoryDataset(IterableDataset):
    """
    Iterable dataset for large-scale trajectory data.
    Lazily loads and yields sequence pairs to avoid memory overflow.
    """

    def __init__(
        self,
        prepared_data_folder: str = "prepared_data",
        patch_size: int = 10,
        max_length: int = 10,
        stride: int = 10,
    ):
        super().__init__()
        self.prepared_data_folder = prepared_data_folder
        self.patch_size = patch_size
        self.max_length = max_length
        self.stride = stride
        self.files = [
            os.path.join(prepared_data_folder, f)
            for f in os.listdir(prepared_data_folder)
            if f.endswith(".parquet")
        ]
        # shuffle the files to ensure randomness
        np.random.shuffle(self.files)

    def _generate_sequences_from_trajectory(self, trajectory):
        patched = trajectory_as_patches(trajectory, patch_size=self.patch_size)
        for i in range(0, len(patched) - self.max_length, self.stride):
            input_seq = torch.tensor(
                patched[i : i + self.max_length], dtype=torch.float32
            )
            output_seq = torch.tensor(
                patched[i + 1 : i + self.max_length + 1], dtype=torch.float32
            )
            yield input_seq, output_seq

    def _process_file(self, file_path):
        table = pq.read_table(file_path)
        trajectories = table.column("trajectory").to_pylist()
        for traj in trajectories:
            yield from self._generate_sequences_from_trajectory(traj)

    @lru_cache
    def __len__(self):
        total_length = 0
        for file_path in self.files:
            table = pq.read_table(file_path)
            trajectories = table.column("trajectory").to_pylist()
            for traj in trajectories:
                for i in range(0, len(traj) - self.max_length, self.stride):
                    total_length += 1

        return total_length

    def __iter__(self):
        for file_path in self.files:
            yield from self._process_file(file_path)


from torch.utils.data import DataLoader


def create_dataloader(
    batch_size=4,
    max_length=256,
    stride=128,
    drop_last=True,
    patch_size=10,
    num_workers=0,
    prepared_data_folder="prepared_data",
    prefetch_factor=2,
) -> DataLoader:
    dataset = TrajectoryDataset(
        prepared_data_folder=prepared_data_folder,
        patch_size=patch_size,
        max_length=max_length,
        stride=stride,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,  # optional but useful for speed
    )

    return dataloader
