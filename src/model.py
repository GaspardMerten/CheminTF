import torch
import torch.nn as nn

from src import config

class TrajectoryEmbedding(nn.Module):
    """
    Embeds each trajectory patch (flattened into a vector of size config.VALUES_PER_POINT * patch_size)
    into a lower-dimensional space.
    """

    def __init__(self, patch_size: int, embedding_dim: int = 4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_size = patch_size
        self.embedding = nn.Linear(config.VALUES_PER_POINT * patch_size, embedding_dim)

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        return self.embedding(trajectory)


class GPTBlock(nn.Module):
    """
    A single transformer block for CheminGPT.
    """

    def __init__(self, embedding_size, n_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embedding_size)
        self.ln2 = nn.LayerNorm(embedding_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=embedding_size, num_heads=n_heads, dropout=dropout
        )
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, 4 * embedding_size),
            nn.GELU(),
            nn.Linear(4 * embedding_size, embedding_size),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        seq_len = x.size(0)
        attn_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf")), diagonal=1
        ).to(x.device)
        attn_output, _ = self.attn(
            self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=attn_mask
        )
        x = x + self.dropout(attn_output)
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x


class CheminGPT(nn.Module):
    """
    GPT-style model for trajectory prediction.
    """

    def __init__(self, patch_size, embedding_size, num_layers, n_heads, max_length):
        super().__init__()
        self.patch_size = patch_size
        self.embedding_size = embedding_size
        self.token_embedding = TrajectoryEmbedding(
            patch_size=patch_size, embedding_dim=embedding_size
        )
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_length, embedding_size))
        self.dropout = nn.Dropout(0.1)
        self.blocks = nn.ModuleList(
            [GPTBlock(embedding_size, n_heads, dropout=0.1) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(embedding_size)
        self.head = nn.Linear(embedding_size, config.VALUES_PER_POINT * patch_size)

    def forward(self, x):
        batch, seq_len, _ = x.size()
        # save last embedding dimension
        last_input_batch = x[:, -1, :].unsqueeze(1)

        # Convert patches to embeddings
        x = self.token_embedding(x.view(-1, self.patch_size * config.VALUES_PER_POINT))
        x = x.view(batch, seq_len, self.embedding_size)
        # Add positional embeddings

        x = x + self.pos_embedding[:, :seq_len, :]
        x = self.dropout(x)
        # Transformer expects shape (seq_len, batch, embedding_size)
        x = x.transpose(0, 1)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        x = x.transpose(0, 1)
        logits = self.head(x)
        # add back the last embedding dimension
        logits += last_input_batch


        return logits
