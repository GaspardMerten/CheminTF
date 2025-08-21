import math
import random
import time
from typing import List, Tuple

from torch.utils.data import Dataset

from new.encoder import trajectory_to_time_annotated_utm_trajectory


class SyntheticTrajectoryGenerator:
    def __init__(
        self,
        start_lat: float = 56.0,
        start_lng: float = 8.0,
        start_timestamp: int = None,
        step_seconds: int = 60,
        num_points: int = 20,
        delta_lat: float = 0.001,
        delta_lng: float = 0.001,
        noise: float = 0.0,
        jitter_time: bool = False,
        seed: int = None,
    ):
        if seed is not None:
            random.seed(seed)
        self.start_lat = start_lat
        self.start_lng = start_lng
        self.step_seconds = step_seconds
        self.num_points = num_points
        self.delta_lat = delta_lat
        self.delta_lng = delta_lng
        self.noise = noise
        self.jitter_time = jitter_time

        if start_timestamp is None:
            import datetime

            start_timestamp = int(
                datetime.datetime.now(datetime.timezone.utc).timestamp()
            )
        self.start_timestamp = start_timestamp

    def generate(self) -> List[Tuple[float, float, int]]:
        lat, lng = self.start_lat, self.start_lng
        timestamp = self.start_timestamp

        trajectory = []

        num_turns = random.randint(1, 5)
        turning_points = sorted(random.sample(range(2, self.num_points - 2), num_turns))

        delta_lat = self.delta_lat
        delta_lng = self.delta_lng

        for i in range(self.num_points):
            # Apply turning point by changing direction randomly
            if i in turning_points:
                angle = random.uniform(-1.5, 1.5)  # radians
                cos_a = math.cos(angle)
                sin_a = math.sin(angle)
                new_delta_lat = cos_a * delta_lat - sin_a * delta_lng
                new_delta_lng = sin_a * delta_lat + cos_a * delta_lng
                delta_lat, delta_lng = new_delta_lat, new_delta_lng

            # Add noise and jitter
            noise_lat = random.uniform(-self.noise, self.noise)
            noise_lng = random.uniform(-self.noise, self.noise)
            time_offset = random.randint(-5, 5) if self.jitter_time else 0

            trajectory.append(
                (lng + noise_lng, lat + noise_lat, timestamp + time_offset)
            )

            lat += delta_lat
            lng += delta_lng
            timestamp += self.step_seconds

        return trajectory


class SyntheticTrajectoryDataset(Dataset):
    def __init__(self, num_trajectories: int, center_lat: float = 56.0, center_lng: float = 8.0):
        self.trajs = []
        self.labels = []

        for _ in range(num_trajectories):
            seed = random.randint(0, int(1e6))
            random.seed(seed)

            # Fully randomized generation parameters
            generator = SyntheticTrajectoryGenerator(
                start_lat=random.uniform(55.5, 56.5),
                start_lng=random.uniform(7.5, 8.5),
                start_timestamp=int(time.time()) + random.randint(-1000, 1000),
                step_seconds=random.choice([30, 60, 120]),
                num_points=random.randint(10, 30),
                delta_lat=random.uniform(-0.01, 0.01),
                delta_lng=random.uniform(-0.01, 0.01),
                noise=random.uniform(0.0, 0.001),
                jitter_time=random.choice([True, False]),
                seed=seed,
            )

            label = 0  # placeholder (you can use directionality or curvature to define labels if needed)

            traj = generator.generate()
            tensor = trajectory_to_time_annotated_utm_trajectory(traj, center_lat, center_lng)

            self.trajs.append(tensor)
            self.labels.append(label)


    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, idx):
        return self.trajs[idx], self.labels[idx]
