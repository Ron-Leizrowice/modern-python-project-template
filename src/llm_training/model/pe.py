"""Tensor dimension notation.

- B: batch size
- T: sequence length
- C: embedding width (d_model)
- L: maximum cached sequence length
"""

import math
from enum import StrEnum

import torch
from torch import nn


class PositionalEncoder(StrEnum):
    """Supported positional encoding strategies."""

    SINUSOIDAL = "sinusoidal"
    LEARNED = "learned"
    ROTARY = "rotary"


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for batch-first inputs.

    Expects ``x`` shaped (B, T, d). Maintains a buffer ``pe`` with shape
    (1, seq_len, d). On each forward pass it slices the first T positions and
    adds them to ``x``.
    """

    pe: torch.Tensor  # (1, max_len, d)

    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))  # (d/2,)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d)

        # Register as a buffer: saved in state_dict, moves with .to(device), no grads.
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positions to embeddings.

        Args:
            x: Tensor with shape (B, T, d).

        Returns:
            Tensor with shape (B, T, d) after adding positions and dropout.
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class LearnedPositionalEncoding(nn.Module):
    """Learned positional embeddings with dropout."""

    def __init__(self, *, max_len: int, d_model: int) -> None:
        super().__init__()
        self.position_embeddings = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device, dtype=torch.long).unsqueeze(0)  # (1, T)
        pos_embed = self.position_embeddings(positions)  # (1, T, C)
        return x + pos_embed  # (B, T, C)


type PositionalEncoding = SinusoidalPositionalEncoding | LearnedPositionalEncoding
