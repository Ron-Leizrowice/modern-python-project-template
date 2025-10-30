import math
from dataclasses import dataclass

import torch

from llm_training.model.gpt import LlmArchitecture, LlmDimensions
from llm_training.model.norm import NormFunction
from llm_training.model.pe import PositionalEncoder
from llm_training.model.transformer_blocks import ActivationFunction
from llm_training.model.weight_init import AttnWeightInitConfig, ModelWeightInit, WeightDistribution, WeightInitConfig
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
    positional_encoding_type: PositionalEncoder = PositionalEncoder.LEARNED
    use_ffn_bias: bool = True
    use_attn_bias: bool = False
    norm: NormFunction = NormFunction.LayerNorm
    tie_encoder_decoder_weights: bool = False
    gradient_checkpointing: bool = True

    ### Weight Init Config ###
    # Embeddings
    embeddings_dtype: torch.dtype = torch.float32
    embedding_weights_init: WeightDistribution = WeightDistribution.NORMAL
    embedding_weights_range: float = 0.0  # only used for UNIFORM
    embedding_weights_std: float = 0.02  # onnly used for NORMAL
    # Positional Encoding
    pe_dtype: torch.dtype = torch.float32
    pe_weights_init: WeightDistribution = WeightDistribution.NORMAL
    pe_weights_range: float = 0.1  # only used for UNIFORM
    pe_weights_std: float = 0.02  # only used for NORMAL
    # Feed-forward Layers
    ff_dtype: torch.dtype = torch.float32
    ff_weights_init: WeightDistribution = WeightDistribution.NORMAL
    ff_weights_range: float = 0.1  # only used for UNIFORM
    ff_weights_std: float = 0.02  # only used for NORMAL
    # Attention Layers
    attn_dtype: torch.dtype = torch.float32
    attn_weights_init: WeightDistribution = WeightDistribution.NORMAL
    attn_weights_range: float = 0.1  # only used for UNIFORM
    attn_weights_std: float = 0.02  # only used for NORMAL
    scale_proj_by_layers: bool = False

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
    batch_size: int = 16  # Micro-batch size per optimizer step before accumulation.")
    grad_accum_steps: int = 4  # Steps over which to accumulate gradients before an update.")
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

    def llm_architecture(self) -> LlmArchitecture:
        return LlmArchitecture(
            activation_function=self.activation_function,
            norm=self.norm,
            positional_encoder=self.positional_encoding_type,
            use_ff_bias=self.use_ffn_bias,
            use_attn_bias=self.use_attn_bias,
            dimensions=self.llm_dimensions(),
            weight_init=self.weight_init(),
            use_gradient_checkpointing=self.gradient_checkpointing,
        )

    def llm_dimensions(self) -> LlmDimensions:
        return LlmDimensions(
            seq_len=self.seq_len,
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_inner=self.d_model * self.d_inner_ratio,
        )

    def weight_init(self) -> ModelWeightInit:
        if self.scale_proj_by_layers:
            proj_scale_factor = math.sqrt(2 * self.n_layers)
        else:
            proj_scale_factor = 1.0

        return ModelWeightInit(
            embedding=WeightInitConfig(
                dtype=self.embeddings_dtype,
                dist=self.embedding_weights_init,
                range=self.embedding_weights_range,
                std=self.embedding_weights_std,
            ),
            pe=WeightInitConfig(
                dtype=self.pe_dtype,
                dist=self.pe_weights_init,
                range=self.pe_weights_range,
                std=self.pe_weights_std,
            ),
            ff=WeightInitConfig(
                dtype=self.ff_dtype,
                dist=self.ff_weights_init,
                range=self.ff_weights_range,
                std=self.ff_weights_std,
            ),
            attn=AttnWeightInitConfig(
                dtype=self.attn_dtype,
                dist=self.attn_weights_init,
                range=self.attn_weights_range,
                std=self.attn_weights_std,
                proj_scale_factor=proj_scale_factor,
            ),
            tie_encoder_decoder_weights=self.tie_encoder_decoder_weights,
        )
