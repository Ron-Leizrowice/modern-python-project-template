import mlx.core as mx
from mlx import nn


class MlxTransformerModel(nn.Module):
    """Decoder-only Transformer implemented with MLX."""

    embedding: nn.Embedding
    positional_encoding: nn.SinusoidalPositionalEncoding
    transformer: nn.TransformerEncoder
    output_projection: nn.Linear

    def __init__(
        self,
        *,
        vocab_size: int,
        num_layers: int,
        dims: int,
        num_heads: int,
        checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dims)
        self.positional_encoding = nn.SinusoidalPositionalEncoding(dims)
        self.transformer = nn.TransformerEncoder(
            num_layers=num_layers,
            dims=dims,
            num_heads=num_heads,
            norm_first=True,
            checkpoint=checkpoint,
        )
        self.output_projection = nn.Linear(dims, vocab_size)

    def __call__(self, tokens: mx.array) -> mx.array:
        """Apply embedding, sinusoidal positions, transformer blocks, and LM head."""
        seq_len = tokens.shape[1]
        causal_mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)

        hidden = self.embedding(tokens)
        hidden = hidden + self.positional_encoding(mx.arange(seq_len))
        hidden = self.transformer(hidden, causal_mask)

        return self.output_projection(hidden)
