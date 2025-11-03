import torch
from torch import nn, Tensor
from src.modules import constants


class OutputModule(nn.Module):
    """
    Output projection module.

    Projects the transformer hidden representations into the final output space.
    """

    def __init__(self, embed_dim: int = constants.EMBEDDING_DIMENSION):
        """
        :param embed_dim: Dimensionality of transformer embeddings before projection.
        """
        super().__init__()
        self.output_proj = nn.Linear(embed_dim, constants.OUTPUT_DIMENSION, dtype=torch.float32)

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: Input tensor of shape [T, B, E], representing hidden states
                  from the transformer encoder, where
                  T = sequence length,
                  B = batch size,
                  E = embedding dimension.
        :return: Tensor of shape [T, B, OUTPUT_DIMENSION], representing projected outputs.
        """
        return self.output_proj(x)
