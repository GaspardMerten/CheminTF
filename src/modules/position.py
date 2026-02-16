import torch
from torch import nn


class PositionalEmbedding(nn.Module):
    """GPT-2 style learnable positional encoding."""

    def __init__(self, embedding_dim: int, max_seq_len: int):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_seq_len, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds learned positional encodings to the input tensor.
        Expects x of shape [T, B, E] or [B, T, E].
        """
        device = x.device
        seq_len = x.size(0) if x.dim() == 3 else x.size(1)
        positions = torch.arange(seq_len, device=device, dtype=torch.long)
        pos_enc = self.position_embeddings(positions)
        if x.shape[0] == seq_len:
            pos_enc = pos_enc.unsqueeze(1)  # [T, 1, E]
        else:
            pos_enc = pos_enc.unsqueeze(0)  # [1, T, E]
        return x + pos_enc
