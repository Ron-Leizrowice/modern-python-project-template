import math
import os
from dataclasses import dataclass
from enum import Enum, StrEnum

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedTokenizerFast

from llm_training_example.tokenizer import get_tokenizer

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"


class FloatPrecision(Enum):
    """Precision types for model training."""

    FP32 = "fp32"
    BF16 = "bf16"
    FP16 = "fp16"

    def to_torch_dtype(self) -> torch.dtype:
        if self is FloatPrecision.FP32:
            return torch.float32
        if self is FloatPrecision.BF16:
            return torch.bfloat16
        if self is FloatPrecision.FP16:
            return torch.float16
        raise ValueError(f"Unsupported FloatPrecision value: {self}.")


class ActivationFunction(StrEnum):
    """Activation function to use in fully connected blocks."""

    RELU = "relu"
    RELU2 = "relu^2"


@dataclass
class LlmDimensions:
    """Configuration parameters for the Transformer model."""

    vocab_size: int  # Tokenizer vocabulary size
    d_model: int  # Transformer hidden dimension
    n_heads: int  # Number of attention heads
    n_layers: int  # Number of Transformer layers
    d_inner: int  # Feed-forward network hidden size


MASK_RANK_BATCHLESS = 2
MASK_RANK_BATCH = 3


class MultiHeadSelfAttention(nn.Module):
    """Manual multi-head self-attention with additive masks."""

    def __init__(self, *, dtype: torch.dtype, d_model: int, n_heads: int, use_bias: bool, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=use_bias, dtype=dtype)
        self.k_proj = nn.Linear(d_model, d_model, bias=use_bias, dtype=dtype)
        self.v_proj = nn.Linear(d_model, d_model, bias=use_bias, dtype=dtype)
        self.out_proj = nn.Linear(d_model, d_model, bias=use_bias, dtype=dtype)

        self.attn_dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        def _merge_heads(tensor: torch.Tensor) -> torch.Tensor:
            tensor = tensor.view(batch_size, seq_len, self.n_heads, self.head_dim)
            return tensor.transpose(1, 2)  # (B, H, T, Hd)

        q = _merge_heads(self.q_proj(x))
        k = _merge_heads(self.k_proj(x))
        v = _merge_heads(self.v_proj(x))

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, T, T)
        neg_inf = torch.finfo(attn_scores.dtype).min

        mask = key_padding_mask.to(dtype=torch.bool, device=attn_scores.device)
        if mask.dim() == MASK_RANK_BATCHLESS:
            # Use ternary operator for mask shape logic.
            mask = mask.unsqueeze(0).unsqueeze(0) if mask.size(0) == mask.size(1) else mask.unsqueeze(1).unsqueeze(2)
        elif mask.dim() == MASK_RANK_BATCH:
            mask = mask.unsqueeze(1)  # (B, 1, T, T)
        else:
            raise ValueError(
                f"Unsupported attention mask rank {mask.dim()}; expected {MASK_RANK_BATCHLESS} or {MASK_RANK_BATCH}."
            )

        attn_scores = attn_scores.masked_fill(~mask, neg_inf)

        attn_weights = torch.softmax(attn_scores, dim=-1).to(attn_scores.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)  # (B, H, T, Hd)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return self.out_proj(attn_output)


def _relu_squared(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x).square()


class FeedForwardNetwork(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(
        self, *, d_model: int, d_inner: int, activation: ActivationFunction, use_bias: bool, dropout: float
    ) -> None:
        super().__init__()
        self.lin_in = nn.Linear(d_model, d_inner, bias=use_bias)
        match activation:
            case ActivationFunction.RELU:
                self.activation = nn.ReLU()
            case ActivationFunction.RELU2:
                self.activation = _relu_squared
        self.lin_out = nn.Linear(d_inner, d_model, bias=use_bias)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin_in(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.lin_out(x)


def _rms_norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


class TransformerBlock(nn.Module):
    """Single Transformer block with residual connections and LayerNorm."""

    def __init__(
        self,
        *,
        dtype: torch.dtype,
        d_model: int,
        n_heads: int,
        d_inner: int,
        activation: ActivationFunction,
        use_ffn_bias: bool,
        use_attn_bias: bool,
        rms_norm: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(
            dtype=dtype,
            d_model=d_model,
            n_heads=n_heads,
            use_bias=use_attn_bias,
            dropout=dropout,
        )
        self.dropout1 = nn.Dropout(p=dropout)
        if rms_norm:
            self.norm1 = _rms_norm
            self.norm2 = _rms_norm
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

        self.ff = FeedForwardNetwork(
            d_model=d_model,
            d_inner=d_inner,
            use_bias=use_ffn_bias,
            activation=activation,
            dropout=dropout,
        )
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        attn_out = self.self_attn(x, key_padding_mask)
        x = self.norm1(x + self.dropout1(attn_out))

        ff_out = self.ff(x)
        return self.norm2(x + self.dropout2(ff_out))


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for batch-first inputs.

    Expects ``x`` shaped (B, T, d). Maintains a buffer ``pe`` with shape
    (1, seq_len, d). On each forward pass it slices the first T positions and
    adds them to ``x``.
    """

    pe: torch.Tensor  # (1, max_len, d)

    def __init__(self, dtype: torch.dtype, d_model: int, dropout: float = 0.1, max_len: int = 4096) -> None:
        super().__init__()
        self.dropout: nn.Dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model, dtype=dtype)
        position = torch.arange(0, max_len, dtype=dtype).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=dtype) * (-math.log(10000.0) / d_model))  # (d/2,)
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
        return self.dropout(x + self.pe[:, :seq_len, :])  # ensure pe built in model dtype


class TransformerModel(nn.Module):
    """Causal Transformer encoder + LM head.

    Dimensions:
        dtype: Floating point precision type to use.
        vocab_size: Vocabulary size V.
        d_model: Embedding/hidden size d.
        n_heads: Number of attention heads.
        d_inner: Feedforward hidden size inside each encoder layer.
        n_layers: Number of stacked encoder layers.
    """

    tokenizer: PreTrainedTokenizerFast

    embeddings: nn.Embedding
    positional_encoder: PositionalEncoding
    layers: nn.ModuleList
    decoder: nn.Linear

    sqrt_d: float

    causal_mask: torch.Tensor

    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        seq_len: int,
        dimensions: LlmDimensions,
        activation_function: ActivationFunction,
        use_ffn_bias: bool,
        use_attn_bias: bool,
        tie_encoder_decoder_weights: bool,
        rms_norm: bool,
        weight_init_range: float,
        dropout: float,
    ) -> None:
        super().__init__()

        self.tokenizer = get_tokenizer(seq_len=seq_len, vocab_size=dimensions.vocab_size)

        self.sqrt_d = math.sqrt(dimensions.d_model)

        # Token embedding table: maps token ids -> d-dimensional vectors.
        self.embeddings = nn.Embedding(
            num_embeddings=dimensions.vocab_size, embedding_dim=dimensions.d_model, dtype=dtype
        )

        # Positional encoding adds deterministic position information to embeddings.
        self.positional_encoder = PositionalEncoding(dtype=dtype, d_model=dimensions.d_model, dropout=dropout)

        self.layers = nn.ModuleList(
            TransformerBlock(
                dtype=dtype,
                d_model=dimensions.d_model,
                n_heads=dimensions.n_heads,
                d_inner=dimensions.d_inner,
                activation=activation_function,
                use_ffn_bias=use_ffn_bias,
                use_attn_bias=use_attn_bias,
                rms_norm=rms_norm,
                dropout=dropout,
            )
            for _ in range(dimensions.n_layers)
        )

        # Final linear head projects hidden states to vocabulary logits (no bias required)
        self.decoder = nn.Linear(in_features=dimensions.d_model, out_features=dimensions.vocab_size, bias=False)
        if tie_encoder_decoder_weights:
            self.decoder.weight = self.embeddings.weight

        self.init_weights(init_range=weight_init_range)

        base_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        self.register_buffer("causal_mask", base_mask, persistent=False)

    @torch.no_grad()
    def init_weights(self, init_range: float) -> None:
        """Simple uniform initialization for embeddings and output head."""
        self.embeddings.weight.uniform_(-init_range, init_range)
        self.decoder.weight.uniform_(-init_range, init_range)

        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.decoder:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.zero_()

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """Apply embedding, positions, Transformer stack, and LM head."""
        x = self._embed_with_positions(src)
        hidden_states = self._run_transformer_blocks(x)
        return self.decoder(hidden_states)

    def _embed_with_positions(self, src: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.embeddings(src) * self.sqrt_d
        return self.positional_encoder(x)

    def _run_transformer_blocks(self, h: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            h = layer(h, self.causal_mask)
        return h


def setup_llm(
    *,
    dtype: torch.dtype,
    device: torch.device,
    dimensions: LlmDimensions,
    activation_function: ActivationFunction,
    tie_encoder_decoder_weights: bool,
    use_ffn_bias: bool,
    use_attn_bias: bool,
    rms_norm: bool,
    weight_init_range: float,
    compile_model: bool,
    seq_len: int,
    dropout: float,
) -> TransformerModel:
    """Instantiate and prepare the Transformer model for training."""
    model = TransformerModel(
        dtype=dtype,
        device=device,
        seq_len=seq_len,
        dimensions=dimensions,
        activation_function=activation_function,
        use_ffn_bias=use_ffn_bias,
        use_attn_bias=use_attn_bias,
        tie_encoder_decoder_weights=tie_encoder_decoder_weights,
        rms_norm=rms_norm,
        dropout=dropout,
        weight_init_range=weight_init_range,
    )

    model = model.to(device=device, dtype=dtype)
    if compile_model:
        model = torch.compile(model)
    assert isinstance(model, TransformerModel)

    return model
