import math
import random
from typing import List, Tuple


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
            enable_turns: bool = True,
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
        self.enable_turns = enable_turns

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

        num_turns = random.randint(1, 5) if self.enable_turns else 0
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
            time_offset = random.randint(1, self.step_seconds - 1) if self.jitter_time else self.step_seconds

            trajectory.append(
                (lng + noise_lng, lat + noise_lat, timestamp + time_offset)
            )

            lat += delta_lat
            lng += delta_lng
            timestamp += time_offset

        return trajectory

