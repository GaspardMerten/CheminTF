import datetime
import math

import torch
import utm

SCALING_FACTOR_X = 400000.0
DELTA_SCALING_FACTOR_X = 1000.0


def spatial_encoding(
    coordinates: torch.Tensor,
) -> torch.Tensor:

    utm_list = [
        utm.from_latlon(lat.item(), lng.item())[:2]
        for lat, lng in coordinates
    ]

    utm_tensor = torch.tensor(utm_list, dtype=torch.float32)

    deltas = torch.zeros_like(utm_tensor)
    deltas[1:] = utm_tensor[1:] - utm_tensor[:-1]

    deltas /= DELTA_SCALING_FACTOR_X
    normalized = utm_tensor / SCALING_FACTOR_X

    return torch.cat([normalized, deltas], dim=1)

def cyclic_encoding(value: torch.Tensor, period: float) -> torch.Tensor:
    """
    Encode a cyclic temporal feature using sine and cosine transformations.

    :param value: Tensor of feature values (e.g., hour of day).
    :param period: The period of the cycle (e.g., 24 for hours in a day).
    :return: Tensor of shape (N, 2) with sine and cosine encodings.
    """

    angle = 2 * math.pi * value / period
    return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)


import torch
import datetime

def temporal_encoding(timestamps: torch.Tensor) -> torch.Tensor:
    """
    Encode timestamps into cyclic temporal features.

    Each timestamp is converted into ten features:
    - Day of week (sin, cos)
    - Hour of day (sin, cos)
    - Minute of hour (sin, cos)
    - Second of minute (sin, cos)
    - Seconds since start
    - Delta seconds from previous timestamp

    :param timestamps: Tensor (N,) — may be on CPU or GPU.
    :return: Tensor (N, 10) on same device as input.
    """
    # ensure dtype is float64 for timestamp precision
    assert timestamps.dtype == torch.float64, "timestamps must be of dtype torch.float64"

    if timestamps.numel() == 0:
        return torch.empty((0, 10), dtype=torch.float32, device=timestamps.device)

    device = timestamps.device
    timestamps = timestamps.to("cpu", dtype=torch.float32)  # datetime requires CPU

    # Convert to datetime objects
    dt_list = [
        datetime.datetime.fromtimestamp(t.item(), tz=datetime.timezone.utc)
        for t in timestamps
    ]

    # Extract calendar components
    day = torch.tensor([dt.weekday() for dt in dt_list], dtype=torch.float32)
    hour = torch.tensor([dt.hour for dt in dt_list], dtype=torch.float32)
    minute = torch.tensor([dt.minute for dt in dt_list], dtype=torch.float32)
    second = torch.tensor([dt.second for dt in dt_list], dtype=torch.float32)

    # Cyclic encodings
    day_enc = cyclic_encoding(day, 7)
    hour_enc = cyclic_encoding(hour, 24)
    minute_enc = cyclic_encoding(minute, 60)
    second_enc = cyclic_encoding(second, 60)

    # Timing features (computed in CPU)
    seconds_since_start = timestamps - timestamps[0]
    delta_seconds = torch.zeros_like(timestamps)
    delta_seconds[1:] = timestamps[1:] - timestamps[:-1]

    # Combine all and move back to original device
    features = torch.cat(
        [
            day_enc,
            hour_enc,
            minute_enc,
            second_enc,
            seconds_since_start.unsqueeze(-1),
            delta_seconds.unsqueeze(-1),
        ],
        dim=1,
    ).to(device)

    # Convert to float32 on the target device
    features = features.to(dtype=torch.float32, device=device)

    return features
