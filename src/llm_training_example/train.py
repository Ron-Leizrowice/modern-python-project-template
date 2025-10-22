"""Core training loop.

The goal of this project is to optimize LLM pre-training performance of a GPT-2 like model.
"""

import math
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import torch
import wandb
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from llm_training_example.constants import DEVICE, WANDB_LOGGING_DIR
from llm_training_example.data import prepare_data
from llm_training_example.model import (
    ActivationFunction,
    LlmDimensions,
    TransformerModel,
    setup_llm,
)
from llm_training_example.optimization import (
    DecaySchedule,
    OptimizerConfig,
    OptimizerType,
    SchedulerConfig,
    WarmupType,
    setup_optimizer_and_scheduler,
)
from llm_training_example.utils import compute_next_token_loss

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

LOSS_AVG_WINDOW = 10  # Number of recent steps to average for loss smoothing.
LOG_INTERVAL = 10


@dataclass(slots=True, frozen=True)
class TrainingConfig:
    """Typed runtime configuration for the training script."""

    wandb_project: str | None  # Weights & Biases project identifier

    # LLM Parameters
    vocab_size: int = 49_152  # Tokenizer vocabulary size
    d_model: int = 64  # Transformer hidden dimension
    n_heads: int = 4  # Number of attention heads
    n_layers: int = 4  # Number of Transformer layers
    d_inner: int = 128  # Feed-forward network hidden size
    activation_function: ActivationFunction = ActivationFunction.RELU
    use_ffn_bias: bool = True
    use_attn_bias: bool = True
    rms_norm: bool = False
    tie_encoder_decoder_weights: bool = False

    dropout: float = 0.0  # Dropout probability applied throughout the model
    weight_init_range: float = 0.1  # Range for uniform weight initialization

    # Optimizer Settings
    lr: float = 2.5e-4  # Base learning rate for the optimizer
    weight_decay: float = 0.01  # Weight decay strength applied by the optimizer
    optimizer_algorithm: OptimizerType = OptimizerType.ADAMW  # Optimizer algorithm to use
    optimizer_eps: float = 1e-8  # Numerical stability epsilon supplied to the optimizer

    # Scheduler Settings
    warmup_type: WarmupType = WarmupType.COSINE  # Warmup schedule
    warmup_ratio: float = 0.1  # Fraction of training dedicated to warmup

    decay_schedule: DecaySchedule = DecaySchedule.COSINE  # Learning rate scheduler type
    min_lr_ratio: float = 0.1  # Relative learning rate maintained after decay completes

    # Training Hyperparameters
    seq_len: int = 512  # Maximum sequence length for training examples.")
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
        )

    def llm_dimensions(self) -> LlmDimensions:
        return LlmDimensions(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_inner=self.d_inner,
        )


class Trainer:
    """Helper class to manage training state."""

    grad_step: int
    total_grad_norm: float

    training_loss: deque[float]

    scheduler: LRScheduler
    optimizer: Optimizer
    max_grad_norm: float

    wandb_run: wandb.Run | None = None

    def setup_wandb(self, config: TrainingConfig) -> None:
        self.wandb_run = wandb.init(project=config.wandb_project, config=asdict(config), dir=WANDB_LOGGING_DIR)

    def setup_trainer(self, model: TransformerModel, dataloader: DataLoader, config: TrainingConfig) -> None:
        assert len(dataloader) % config.grad_accum_steps == 0
        self.total_steps = len(dataloader) // config.grad_accum_steps

        self.grad_step = 0
        self.total_grad_norm = 0.0
        self.training_loss = deque(maxlen=LOSS_AVG_WINDOW)

        self.vocab_size = config.vocab_size
        self.trainable_params = [param for param in model.parameters() if param.requires_grad]
        self.optimizer, self.scheduler = setup_optimizer_and_scheduler(
            model, config.optimizer(), config.scheduler(), self.total_steps
        )
        self.scaler = torch.GradScaler()

        self.max_grad_norm = config.clip_grad_norm

        if config.wandb_project:
            self.setup_wandb(config)

    def update_gradient(self) -> None:
        self.total_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.trainable_params,
            self.max_grad_norm,
            norm_type=2.0,
        ).item()

        # Unscales and applies gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()  # Updates scaling factor as necessary

        # Clear gradients for the next iteration
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()

        self.grad_step = 0

    def train(self, model: TransformerModel, dataloader: DataLoader, config: TrainingConfig) -> None:
        """Main training loop for the Transformer language model."""

        self.setup_trainer(model, dataloader, config)

        model.train()
        self.optimizer.zero_grad(set_to_none=True)

        progress_bar: Any = tqdm(total=self.total_steps, desc="Training", unit="batch", dynamic_ncols=True)
        with progress_bar:
            for step, inputs in enumerate(dataloader):
                batch_inputs = inputs[0].to(device=DEVICE, non_blocking=True)

                # Forward pass with autocast enabled for mixed precision
                with torch.autocast(device_type="cuda"):
                    logits = model(batch_inputs)
                    loss = compute_next_token_loss(logits, batch_inputs)

                train_loss = loss.detach().item()
                if math.isnan(train_loss):
                    raise ValueError("NaN loss encountered")
                self.training_loss.append(train_loss)
                loss /= config.grad_accum_steps

                # Scales gradients to prevent underflow
                self.scaler.scale(loss).backward()

                self.grad_step += 1
                if self.grad_step == config.grad_accum_steps:
                    self.update_gradient()

                    current_lr = self.optimizer.param_groups[0]["lr"]
                    progress_bar.update(1)
                    progress_bar.set_postfix(
                        {
                            "loss": f"{sum(self.training_loss) / min(step + 1, LOSS_AVG_WINDOW):.4f}",
                            "lr": f"{current_lr:.2e}",
                        },
                        refresh=False,
                    )
                    progress_bar.refresh()
                    if self.wandb_run:
                        self.wandb_run.log(
                            data={
                                "train/loss": train_loss,
                                "train/lr": current_lr,
                                "train/total_grad_norm": self.total_grad_norm,
                            },
                            step=step + 1,
                        )

        # Finish the wandb run
        if self.wandb_run:
            self.wandb_run.finish()


def main() -> None:
    """Entry point for training the Transformer language model."""
    config = TrainingConfig(wandb_project=None)

    # Setup model
    model = setup_llm(
        dimensions=config.llm_dimensions(),
        activation_function=config.activation_function,
        use_attn_bias=config.use_attn_bias,
        use_ffn_bias=config.use_ffn_bias,
        tie_encoder_decoder_weights=config.tie_encoder_decoder_weights,
        rms_norm=config.rms_norm,
        weight_init_range=config.weight_init_range,
        seq_len=config.seq_len,
        dropout=config.dropout,
    )

    dataloader = prepare_data(
        tokenizer=model.tokenizer,
        batch_size=config.batch_size,
        grad_accum_steps=config.grad_accum_steps,
    )

    trainer = Trainer()
    trainer.train(model, dataloader, config)


if __name__ == "__main__":
    main()
