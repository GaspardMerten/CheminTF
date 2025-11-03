import torch
from src.modules.features import extract_spatial_features, extract_temporal_features

@torch.no_grad()
def predict_autoregressive(
    model,
    init_coords: torch.Tensor,   # [T₀, 2]
    init_times: torch.Tensor,    # [T₀]
    num_future_steps: int,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Autoregressively predicts future trajectory points given an initial seed.
    """
    model.eval()
    coords = init_coords.clone().to(device)
    times = init_times.clone().to(device)

    for step in range(num_future_steps):
        # 1️⃣ Extract features for current sequence
        spatial_feats = extract_spatial_features(coords).unsqueeze(1).to(device)
        temporal_feats = extract_temporal_features(times).unsqueeze(1).to(device)

        # max 49 points for the model
        if spatial_feats.shape[0] > 49:
            spatial_feats = spatial_feats[-49:, :, :]
            temporal_feats = temporal_feats[-49:, :, :]

        # 2️⃣ Predict next-step delta
        preds = model(spatial_feats, temporal_feats)  # [T, 1, D_out]
        next_delta = preds[-1, 0]                     # [D_out] (still on device)

        dlat, dlng, dt = next_delta
        dlat, dlng = dlat / 1000.0, dlng / 1000.0  # scale back
        dlat = dlat
        dlng = dlng
        dt = dt * 60.0                         # scale back to seconds
        next_t = times[-1] + dt
        # 3️⃣ Append predicted point (everything stays on device)
        next_coord = coords[-1] + torch.tensor([dlat, dlng], device=device)
        coords = torch.cat([coords, next_coord.unsqueeze(0)], dim=0)
        times = torch.cat([times, next_t.unsqueeze(0)], dim=0)


    return coords, times
