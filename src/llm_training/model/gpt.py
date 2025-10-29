import math
from dataclasses import dataclass

import torch
from torch import nn
from transformers import PreTrainedTokenizerFast

from llm_training.constants import DEVICE
from llm_training.model.activations import ActivationFunction
from llm_training.model.blocks import TransformerBlock
from llm_training.model.norm import _rms_norm
from llm_training.model.pe import LearnedPositionalEncoding, PositionalEncoding, PositionalEncodingType
from llm_training.model.weight_init import (
    ModelWeightInit,
    init_embedding,
    init_positional_encoding,
    init_transformer_block,
)
from llm_training.tokenizer import get_tokenizer


@dataclass
class LlmDimensions:
    """Configuration parameters for the Transformer model."""

    vocab_size: int  # Tokenizer vocabulary size
    d_model: int  # Transformer hidden dimension
    n_heads: int  # Number of attention heads
    n_layers: int  # Number of Transformer layers
    d_inner: int  # Feed-forward network hidden size


class TransformerModel(nn.Module):
    """Causal Transformer encoder + LM head.

    Dimensions:
        vocab_size: Vocabulary size V.
        d_model: Embedding/hidden size d.
        n_heads: Number of attention heads.
        d_inner: Feedforward hidden size inside each encoder layer.
        n_layers: Number of stacked encoder layers.
    """

    tokenizer: PreTrainedTokenizerFast
    causal_mask: torch.Tensor

    embeddings: nn.Embedding
    positional_encoder: nn.Module
    blocks: nn.ModuleList
    lm_head: nn.Linear

    def __init__(
        self,
        *,
        seq_len: int,
        dimensions: LlmDimensions,
        activation_function: ActivationFunction,
        use_ffn_bias: bool,
        use_attn_bias: bool,
        tie_encoder_decoder_weights: bool,
        rms_norm: bool,
        positional_encoding_type: PositionalEncodingType,
        weight_init: ModelWeightInit,
    ) -> None:
        super().__init__()

        self.tokenizer = get_tokenizer(seq_len=seq_len, vocab_size=dimensions.vocab_size)

        # Token embedding table: maps token ids -> d-dimensional vectors.
        self.embeddings = nn.Embedding(num_embeddings=dimensions.vocab_size, embedding_dim=dimensions.d_model)

        # Positional encoding adds deterministic position information to embeddings.
        if positional_encoding_type is PositionalEncodingType.LEARNED:
            self.positional_encoder: nn.Module = LearnedPositionalEncoding(
                max_len=seq_len,
                d_model=dimensions.d_model,
            )
        else:
            self.positional_encoder = PositionalEncoding(d_model=dimensions.d_model, max_len=seq_len)

        self.blocks = nn.ModuleList(
            TransformerBlock(
                d_model=dimensions.d_model,
                n_heads=dimensions.n_heads,
                d_inner=dimensions.d_inner,
                activation=activation_function,
                use_ffn_bias=use_ffn_bias,
                use_attn_bias=use_attn_bias,
                rms_norm=rms_norm,
            )
            for _ in range(dimensions.n_layers)
        )

        # Final linear head projects hidden states to vocabulary logits (no bias required)
        self.lm_head = nn.Linear(in_features=dimensions.d_model, out_features=dimensions.vocab_size, bias=False)
        if tie_encoder_decoder_weights:
            self.lm_head.weight = self.embeddings.weight

        if rms_norm:
            self.final_norm = _rms_norm
        else:
            self.final_norm = nn.LayerNorm(dimensions.d_model)

        self.init_weights(config=weight_init)

        causal_mask = torch.ones(seq_len, seq_len, dtype=torch.bool).triu(1)
        self.register_buffer("causal_mask", causal_mask, persistent=False)

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

        # If the positional encodings are learnable, init them
        init_positional_encoding(self.positional_encoder, config=config.pe)

        for block in self.blocks:
            init_transformer_block(block, config.ff, config.attn)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """Apply embedding, positions, Transformer stack, and LM head."""
        x = self._embed_with_positions(src)
        hidden_states = self._run_transformer_blocks(x.to(self.ff_dtype))
        hidden_states = self.final_norm(hidden_states)
        return self.lm_head(hidden_states.to(self.embeddings_dtype))

    def _embed_with_positions(self, src: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.embeddings(src)
        return self.positional_encoder(x.to(self.pe_dtype))

    def _run_transformer_blocks(self, h: torch.Tensor) -> torch.Tensor:
        seq_len = h.size(1)
        causal_mask = self.causal_mask[:seq_len, :seq_len].unsqueeze(0).unsqueeze(1)  # (1,1,T,T)
        for layer in self.blocks:
            h = layer(h, causal_mask)
        return h


def setup_llm(
    *,
    dimensions: LlmDimensions,
    activation_function: ActivationFunction,
    tie_encoder_decoder_weights: bool,
    use_ffn_bias: bool,
    use_attn_bias: bool,
    rms_norm: bool,
    seq_len: int,
    positional_encoding_type: PositionalEncodingType,
    weight_init: ModelWeightInit,
) -> TransformerModel:
    """Instantiate and prepare the Transformer model for training."""
    model = TransformerModel(
        seq_len=seq_len,
        dimensions=dimensions,
        activation_function=activation_function,
        use_ffn_bias=use_ffn_bias,
        use_attn_bias=use_attn_bias,
        tie_encoder_decoder_weights=tie_encoder_decoder_weights,
        rms_norm=rms_norm,
        positional_encoding_type=positional_encoding_type,
        weight_init=weight_init,
    )

    model = model.to(device=DEVICE)
    model.compile(
        fullgraph=True,
        dynamic=False,
    )
    return model
