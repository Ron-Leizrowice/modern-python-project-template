"""Tensor dimension notation.

- B: batch size
- T: sequence length
- V: vocabulary size
- C: model width (d_model)
- I: feed-forward hidden width (d_inner)
- H: attention heads
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from llm_training.model.norm import FunctionalRmsNorm, NormFunction
from llm_training.model.pe import (
    LearnedPositionalEncoding,
    PositionalEncoder,
    PositionalEncoding,
    SinusoidalPositionalEncoding,
)
from llm_training.model.transformer_blocks import ActivationFunction, TransformerBlock
from llm_training.model.weight_init import (
    ModelWeightInit,
    init_embedding,
    init_positional_encoding,
    init_transformer_block,
)


@dataclass
class LlmDimensions:
    """Configuration parameters for the Transformer model."""

    seq_len: int  # Maximum input sequence length
    vocab_size: int  # Tokenizer vocabulary size
    d_model: int  # Transformer hidden dimension
    d_inner: int  # Feed-forward network hidden size
    n_heads: int  # Number of attention heads
    n_layers: int  # Number of Transformer layers


@dataclass
class LlmArchitecture:
    """Configuration for Transformer model architecture."""

    activation_function: ActivationFunction
    norm: NormFunction
    positional_encoder: PositionalEncoder
    use_ff_bias: bool
    use_attn_bias: bool
    dimensions: LlmDimensions
    weight_init: ModelWeightInit
    use_gradient_checkpointing: bool


class TransformerModel(nn.Module):
    """Causal Transformer encoder + LM head.

    Dimensions:
        vocab_size: Vocabulary size V.
        d_model: Embedding/hidden size d.
        n_heads: Number of attention heads.
        d_inner: Feedforward hidden size inside each encoder layer.
        n_layers: Number of stacked encoder layers.
    """

    embeddings: nn.Embedding
    embedding_norm: nn.Module
    positional_encoder: PositionalEncoding
    blocks: nn.ModuleList
    output_norm: nn.Module
    lm_head: nn.Linear

    def __init__(self, config: LlmArchitecture) -> None:
        super().__init__()

        dimensions = config.dimensions

        # Token embedding table: maps token ids -> (C)
        self.embeddings = nn.Embedding(num_embeddings=dimensions.vocab_size, embedding_dim=dimensions.d_model)

        # Positional encoder returns tensor shaped (B, T, C)
        match config.positional_encoder:
            case PositionalEncoder.SINUSOIDAL:
                self.positional_encoder = SinusoidalPositionalEncoding(
                    d_model=dimensions.d_model, max_len=dimensions.seq_len
                )
            case PositionalEncoder.LEARNED:
                self.positional_encoder = LearnedPositionalEncoding(
                    max_len=dimensions.seq_len,
                    d_model=dimensions.d_model,
                )

        self.blocks = nn.ModuleList(
            TransformerBlock(
                d_model=dimensions.d_model,
                n_heads=dimensions.n_heads,
                d_inner=dimensions.d_inner,
                activation=config.activation_function,
                use_ffn_bias=config.use_ff_bias,
                use_attn_bias=config.use_attn_bias,
                norm=config.norm,
            )
            for _ in range(dimensions.n_layers)
        )
        self.use_gradient_checkpointing = config.use_gradient_checkpointing

        # Final linear head projects (B, T, C) -> (B, T, V)
        self.lm_head = nn.Linear(in_features=dimensions.d_model, out_features=dimensions.vocab_size, bias=False)

        match config.norm:
            case NormFunction.FunctionalRmsNorm:
                self.embedding_norm = FunctionalRmsNorm(dimensions.d_model)
                self.output_norm = FunctionalRmsNorm(dimensions.d_model)
            case NormFunction.LayerNorm:
                self.embedding_norm = nn.LayerNorm(dimensions.d_model)
                self.output_norm = nn.LayerNorm(dimensions.d_model)

        self.init_weights(config=config.weight_init)

    @torch.no_grad()
    def init_weights(self, config: ModelWeightInit) -> None:
        """Simple uniform initialization for embeddings and output head."""

        self.embeddings_dtype = config.embedding.dtype
        self.pe_dtype = config.pe.dtype
        self.ff_dtype = config.ff.dtype
        self.attn_dtype = config.attn.dtype

        # Init the Embeddings + LM-head
        init_embedding(self.embeddings, config=config.embedding)
        init_embedding(self.lm_head, config=config.embedding)
        if config.tie_encoder_decoder_weights:
            self.lm_head.weight = self.embeddings.weight

        # If the positional encodings are learnable, init them
        init_positional_encoding(self.positional_encoder, config=config.pe)

        for block in self.blocks:
            init_transformer_block(block, config.ff, config.attn)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """Apply embedding, positions, Transformer stack, and LM head."""
        wte = self._embed_with_positions(src)
        h = self._run_transformer_blocks(wte)
        logits = self._generate_logits(h)
        return logits

    def _embed_with_positions(self, src: torch.Tensor) -> torch.Tensor:
        # Generate embeddings from token indices -> (B, T, C)
        wte = self.embeddings(src)
        # Normalize token level embeddings -> (B, T, C)
        wte = self.embedding_norm(wte)
        # Cast to positional-encoder dtype
        wte = wte.to(self.pe_dtype)
        # Apply positional embeddings -> (B, T, C)
        return self.positional_encoder(wte)

    def _run_transformer_blocks(self, h: torch.Tensor) -> torch.Tensor:
        # Cast to transformer block dtype
        h = h.to(self.ff_dtype)
        # Pass through each transformer layer; optionally checkpoint to trade compute for memory
        for layer in self.blocks:
            if self.use_gradient_checkpointing and self.training:
                h = checkpoint(layer, h, use_reentrant=False)  # type: ignore
            else:
                h = layer(h)  # (B, T, C)
        return h

    def _generate_logits(self, h: torch.Tensor) -> torch.Tensor:
        # Cast to lm-head dtype
        h = h.to(self.embeddings_dtype)
        # Apply output norm -> (B, T, C)
        h = self.output_norm(h)
        # Project hidden state to obtain logits -> (B, T, V)
        return self.lm_head(h)
