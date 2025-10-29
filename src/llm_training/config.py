from dataclasses import dataclass

import torch

from llm_training.model.activations import ActivationFunction
from llm_training.model.gpt import LlmDimensions
from llm_training.model.pe import PositionalEncodingType
from llm_training.model.weight_init import ModelWeightInit, WeightInitConfig, WeightInitType
from llm_training.optimization import (
    DecaySchedule,
    OptimizerConfig,
    OptimizerType,
    SchedulerConfig,
    WarmupType,
)


@dataclass(slots=True)
class TrainingConfig:
    """Typed runtime configuration for the training script."""

    wandb_project: str | None  # Weights & Biases project identifier

    # LLM Parameters
    vocab_size: int = 49_152  # Tokenizer vocabulary size
    d_model: int = 512  # Transformer hidden dimension
    n_heads: int = 4  # Number of attention heads
    n_layers: int = 8  # Number of Transformer layers
    d_inner_ratio: int = 4  # Feed-forward network hidden size
    activation_function: ActivationFunction = ActivationFunction.RELU
    positional_encoding_type: PositionalEncodingType = PositionalEncodingType.SINUSOIDAL
    use_ffn_bias: bool = True
    use_attn_bias: bool = True
    rms_norm: bool = False
    tie_encoder_decoder_weights: bool = True

    ### Weight Init Config ###
    # Embeddings
    embeddings_dtype: torch.dtype = torch.bfloat16
    embedding_weights_init: WeightInitType = WeightInitType.NORMAL
    embedding_weights_range: float = 0.0
    embedding_weights_std: float = 1.0
    # Positional Encoding
    pe_dtype: torch.dtype = torch.float32
    pe_weights_init: WeightInitType = WeightInitType.SPECTRAL
    pe_weights_range: float = 0.0
    pe_weights_std: float = 0.05
    # Feed-forward Layers
    ff_dtype: torch.dtype = torch.float32
    ff_weights_init: WeightInitType = WeightInitType.SPECTRAL
    ff_weights_range: float = 0.2
    ff_weights_std: float = 0.05
    # Attention Layers
    attn_dtype: torch.dtype = torch.float32
    attn_weights_init: WeightInitType = WeightInitType.SPECTRAL
    attn_weights_range: float = 0.2
    attn_weights_std: float = 0.05

    # Optimizer Settings
    lr: float = 1e-3  # Base learning rate for the optimizer
    weight_decay: float = 0.01  # Weight decay strength applied by the optimizer
    optimizer_algorithm: OptimizerType = OptimizerType.ADAMW  # Optimizer algorithm to use
    optimizer_eps: float = 1e-10  # Numerical stability epsilon supplied to the optimizer
    beta1: float = 0.9
    beta2: float = 0.999

    # Scheduler Settings
    warmup_type: WarmupType = WarmupType.COSINE  # Warmup schedule
    warmup_ratio: float = 0.1  # Fraction of training dedicated to warmup

    decay_schedule: DecaySchedule = DecaySchedule.COSINE  # Learning rate scheduler type
    min_lr_ratio: float = 0.1  # Relative learning rate maintained after decay completes

    # Training Hyperparameters
    seq_len: int = 1024  # Maximum sequence length for training examples.")
    batch_size: int = 8  # Micro-batch size per optimizer step before accumulation.")
    grad_accum_steps: int = 8  # Steps over which to accumulate gradients before an update.")
    clip_grad_norm: float = 1.0  # Max L2 norm for global gradient clipping.")

    def scheduler(self) -> SchedulerConfig:
        return SchedulerConfig(
            warmup_type=self.warmup_type,
            warmup_ratio=self.warmup_ratio,
            decay=self.decay_schedule,
            min_lr_ratio=self.min_lr_ratio,
        )

    def optimizer(self) -> OptimizerConfig:
        return OptimizerConfig(
            algorithm=self.optimizer_algorithm,
            lr=self.lr,
            weight_decay=self.weight_decay,
            eps=self.optimizer_eps,
            beta1=self.beta1,
            beta2=self.beta2,
        )

    def llm_dimensions(self) -> LlmDimensions:
        return LlmDimensions(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_inner=self.d_model * self.d_inner_ratio,
        )

    def weight_init(self) -> ModelWeightInit:
        return ModelWeightInit(
            embedding=WeightInitConfig(
                dtype=self.embeddings_dtype,
                type=self.embedding_weights_init,
                range=self.embedding_weights_range,
                std=self.embedding_weights_std,
            ),
            pe=WeightInitConfig(
                dtype=self.pe_dtype,
                type=self.pe_weights_init,
                range=self.pe_weights_range,
                std=self.pe_weights_std,
            ),
            ff=WeightInitConfig(
                dtype=self.ff_dtype,
                type=self.ff_weights_init,
                range=self.ff_weights_range,
                std=self.ff_weights_std,
            ),
            attn=WeightInitConfig(
                dtype=self.attn_dtype,
                type=self.attn_weights_init,
                range=self.attn_weights_range,
                std=self.attn_weights_std,
            ),
        )
