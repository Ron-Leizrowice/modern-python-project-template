import torch
import torch.nn.functional as F
from torch import nn

from llm_training.model.activations import ActivationFunction, _gelu, _relu, _relu_squared
from llm_training.model.norm import _rms_norm


class MultiHeadSelfAttention(nn.Module):
    """Manual multi-head self-attention with additive masks."""

    def __init__(self, *, d_model: int, n_heads: int, use_bias: bool) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=use_bias)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, T, T)

        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)

        attn_output = torch.matmul(attn_weights, v)  # (B, H, T, Hd)
        attn_output = attn_output.transpose(1, 2).contiguous()  # (B, T, H, Hd)
        attn_output = attn_output.view(B, T, C)  # (B, T, d_model)

        return self.out_proj(attn_output)


class FeedForwardNetwork(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, *, d_model: int, d_inner: int, activation: ActivationFunction, use_bias: bool) -> None:
        super().__init__()
        self.fc = nn.Linear(d_model, d_inner, bias=use_bias)
        self.proj = nn.Linear(d_inner, d_model, bias=use_bias)
        match activation:
            case ActivationFunction.RELU:
                self.activation = _relu
            case ActivationFunction.RELU2:
                self.activation = _relu_squared
            case ActivationFunction.GELU:
                self.activation = _gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.activation(x)
        return self.proj(x)


class TransformerBlock(nn.Module):
    """Single Transformer block with residual connections and LayerNorm."""

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        d_inner: int,
        activation: ActivationFunction,
        use_ffn_bias: bool,
        use_attn_bias: bool,
        rms_norm: bool,
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

        if rms_norm:
            self.norm1 = _rms_norm
            self.norm2 = _rms_norm
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), causal_mask)
        return x + self.ff(self.norm2(x))
