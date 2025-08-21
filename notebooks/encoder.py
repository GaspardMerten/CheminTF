import datetime
from typing import List, Tuple

import torch
import utm
from torch import Tensor

# Constants for normalization of the spatial coordinates
SX = SY = 4000

def extract_temporal_features(t: int, t0: int) -> List[float]:
    dt = datetime.datetime.fromtimestamp(t, tz=datetime.timezone.utc)
    return [
        dt.weekday() / 6,
        dt.hour / 23,
        dt.minute / 59,
        ((t - t0) % 3600) / 3600
    ]


def trajectory_to_time_annotated_utm_trajectory(
    trajectory: List[Tuple[float, float, int]], center_lat: float, center_lng: float, zone_number: int, zone_letter: str
) -> Tensor:

    annotated_trajectory = []

    utm_center_x, utm_center_y, _, _ = utm.from_latlon(center_lat, center_lng, zone_number, zone_letter)
    t0 = trajectory[0][2]

    for lng, lat, timestamp in trajectory:
        utm_x, utm_y, _, _ = utm.from_latlon(lat, lng)
        norm_x = (utm_x - utm_center_x) / SX
        norm_y = (utm_y - utm_center_y) / SY

        temporal_features = extract_temporal_features(timestamp, t0)
        annotated_trajectory.append(
            [
                norm_x,
                norm_y,
            ]
            + temporal_features
        )

    return torch.tensor(annotated_trajectory, dtype=torch.float32)

def time_annotated_utm_trajectory_to_trajectory(
    trajectory: Tensor, center_lat: float, center_lng: float, zone_number: int, zone_letter: str
) -> List[Tuple[float, float, int]]:
    """
    Convert a time-annotated UTM trajectory back to a list of (longitude, latitude, timestamp) tuples.
    """
    utm_center_x, utm_center_y, _, _ = utm.from_latlon(center_lat, center_lng)
    t0 = trajectory[0][2] * 86400  # Convert to seconds of the day

    result = []
    for i in range(trajectory.shape[0]):
        norm_x = trajectory[i][0].item() * SX + utm_center_x
        norm_y = trajectory[i][1].item() * SY + utm_center_y
        timestamp = int(t0 + trajectory[i][2] * 3600 + trajectory[i][3] * 60 + trajectory[i][4]*3600)
        lat, lng = utm.to_latlon(norm_x, norm_y, zone_number, zone_letter)
        result.append((lng, lat, timestamp))

    return result