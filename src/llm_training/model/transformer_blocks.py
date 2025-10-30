"""Tensor dimension notation.

- B: batch size
- T: sequence length
- C: model width (d_model)
- H: attention heads
- Hd: per-head width (C / H)
- I: feed-forward hidden width (d_inner)
"""

import torch
import torch.nn.functional as F
from torch import nn

from llm_training.model.activations import ActivationFunction
from llm_training.model.norm import FunctionalRmsNorm, NormFunction


class MultiHeadSelfAttention(nn.Module):
    """Manual multi-head self-attention with additive masks."""

    w_q: nn.Linear
    w_k: nn.Linear
    w_v: nn.Linear
    w_proj: nn.Linear

    def __init__(self, *, d_model: int, n_heads: int, use_bias: bool) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=use_bias)
        self.w_k = nn.Linear(d_model, d_model, bias=use_bias)
        self.w_v = nn.Linear(d_model, d_model, bias=use_bias)
        self.w_proj = nn.Linear(d_model, d_model, bias=use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.size()

        # Project to multi-head tensors with shape (B, H, T, Hd)
        q = self.w_q(x).view(B, T, self.n_heads, self.head_dim)
        k = self.w_k(x).view(B, T, self.n_heads, self.head_dim)
        v = self.w_v(x).view(B, T, self.n_heads, self.head_dim)

        # Run SDPA
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_output = attn_output.transpose(1, 2).flatten(-2)

        # Apply output projection
        attn_output = self.w_proj(attn_output)

        return attn_output


class FeedForwardNetwork(nn.Module):
    """Position-wise feed-forward network."""

    activation: nn.Module

    def __init__(self, *, d_model: int, d_inner: int, activation: ActivationFunction, use_bias: bool) -> None:
        super().__init__()
        # Gated activation functions halve the output dimension, so we need to widen first
        d_proj = d_inner * activation.width_multiplier
        self.fc = nn.Linear(d_model, d_proj, bias=use_bias)

        self.proj = nn.Linear(d_inner, d_model, bias=use_bias)
        self.activation = activation.module()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)  # (B, T, I)
        x = self.activation(x)  # (B, T, I)
        return self.proj(x)  # (B, T, C)


class TransformerBlock(nn.Module):
    """Single Transformer block with residual connections and LayerNorm."""

    norm1: nn.Module
    norm2: nn.Module

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        d_inner: int,
        activation: ActivationFunction,
        norm: NormFunction,
        use_ffn_bias: bool,
        use_attn_bias: bool,
    ) -> None:
        super().__init__()
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            use_bias=use_attn_bias,
        )
        self.ff = FeedForwardNetwork(
            d_model=d_model,
            d_inner=d_inner,
            use_bias=use_ffn_bias,
            activation=activation,
        )

        match norm:
            case NormFunction.FunctionalRmsNorm:
                self.norm1 = FunctionalRmsNorm(d_model)
                self.norm2 = FunctionalRmsNorm(d_model)
            case NormFunction.LayerNorm:
                self.norm1 = nn.LayerNorm(d_model)
                self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))  # Residual over (B, T, C)
        return x + self.ff(self.norm2(x))  # (B, T, C)
