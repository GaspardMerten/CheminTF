import torch
from torch import nn
from src.modules import constants


class SpatioTemporalEmbeddings(nn.Module):
    """
    Projects encoded spatial and temporal features into a combined embedding,
    with built-in normalization to align feature scales.

    Each feature dimension is standardized across the batch and time axes
    before linear projection, ensuring consistent magnitude between
    spatial (lat/lng) and temporal (timestamps or intervals) inputs.
    """

    def __init__(
        self,
        spatial_encoding_dimension: int = constants.SPATIAL_ENCODING_DIMENSION,
        temporal_encoding_dimension: int = constants.TEMPORAL_ENCODING_DIMENSION,
        spatial_embedding_dimension: int = constants.SPATIAL_EMBEDDING_DIMENSION,
        temporal_embedding_dimension: int = constants.TEMPORAL_EMBEDDING_DIMENSION,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps
        self.spatial_fc = nn.Linear(
            spatial_encoding_dimension, spatial_embedding_dimension, dtype=torch.float32
        )
        self.temporal_fc = nn.Linear(
            temporal_encoding_dimension,
            temporal_embedding_dimension,
            dtype=torch.float32,
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize each feature dimension to zero mean and unit variance."""
        mean = x.mean(dim=(0, 1), keepdim=True)
        std = x.std(dim=(0, 1), keepdim=True)
        return (x - mean) / (std + self.eps)

    def forward(
        self, spatial_input: torch.Tensor, temporal_input: torch.Tensor
    ) -> torch.Tensor:
        # spatial_input, temporal_input: [B, T, D]
        spatial_normed = self._normalize(spatial_input)
        temporal_normed = self._normalize(temporal_input)

        spatial_embed = self.spatial_fc(spatial_normed)
        temporal_embed = self.temporal_fc(temporal_normed)

        return torch.cat(
            [spatial_embed, temporal_embed], dim=-1
        )  # [B, T, spatial_emb + temporal_emb]
