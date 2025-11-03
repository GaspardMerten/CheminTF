import torch
from torch import nn, Tensor

from src.modules import constants
from src.modules.encoder import SpatioTemporalEncoder
from src.modules.output import OutputModule
from src.modules.position import LearnedPositionalEncoding


def causal_mask(seq_len: int) -> Tensor:
    # Generates an upper-triangular matrix filled with -inf above the diagonal
    # Used to mask future tokens in causal (autoregressive) transformers
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, float("-inf"))
    return mask


class CheminTF(nn.Module):
    def __init__(
        self,
        n_heads: int,
        num_layers: int,
        embed_dim: int = constants.EMBEDDING_DIMENSION,
        max_len: int = 50,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_encoder = SpatioTemporalEncoder(
            spatial_features_dimension=constants.SPATIAL_FEATURES_DIMENSION,
            temporal_features_dimension=constants.TEMPORAL_FEATURES_DIMENSION,
            spatial_embedding_dimension=constants.SPATIAL_EMBEDDING_DIMENSION,
            temporal_embedding_dimension=constants.TEMPORAL_EMBEDDING_DIMENSION,
        )

        self.pos_encoder = LearnedPositionalEncoding(
            embedding_dim=constants.EMBEDDING_DIMENSION, max_seq_len=max_len
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            # activation => we want some non-linearity here but we want to keep negative values
            activation="gelu",
            nhead=n_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            batch_first=False,
            dtype=torch.float32,
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output = OutputModule(embed_dim=embed_dim)


    def forward(self, spatial_input: torch.Tensor, temporal_input: torch.Tensor):
        # x: [T, B, D]
        x = self.input_encoder(spatial_input, temporal_input)
        # x: [T, B, D]
        T, B, D = x.shape
        x = self.pos_encoder(x)  # [T, B, E]
        mask = causal_mask(T).to(x.device)  # [T, T]

        x = self.transformer(x, mask=mask)  # [T, B, E]

        return self.output(x)
