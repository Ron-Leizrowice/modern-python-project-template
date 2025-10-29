"""Core training loop.

The goal of this project is to optimize LLM pre-training performance of a GPT-2 like model.
"""

import math
import sys
from dataclasses import asdict
from typing import Any

import click
import torch
import wandb
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from llm_training.config import TrainingConfig
from llm_training.constants import DEVICE, WANDB_LOGGING_DIR
from llm_training.data import prepare_data
from llm_training.model.gpt import (
    TransformerModel,
    setup_llm,
)
from llm_training.optimization import (
    setup_optimizer_and_scheduler,
)
from llm_training.utils import compute_next_token_loss

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.enable_flash_sdp(enabled=True)
torch.backends.cuda.enable_mem_efficient_sdp(enabled=True)
torch.backends.cuda.enable_math_sdp(enabled=False)


class Trainer:
    """Helper class to manage training state."""

    grad_step: int
    total_grad_norm: float
    grad_update_steps: int

    batch_loss: float

    scheduler: LRScheduler
    optimizer: Optimizer
    max_grad_norm: float

    wandb_run: wandb.Run | None = None

    def setup_wandb(self, config: TrainingConfig) -> None:
        self.wandb_run = wandb.init(project=config.wandb_project, config=asdict(config), dir=WANDB_LOGGING_DIR)

    def setup_trainer(
        self, model: TransformerModel, dataloader: DataLoader, config: TrainingConfig, *, overfit: bool = False
    ) -> None:
        assert len(dataloader) % config.grad_accum_steps == 0
        if not overfit:
            self.total_steps = len(dataloader) // config.grad_accum_steps

        self.grad_step = 0
        self.total_grad_norm = 0.0
        self.batch_loss = 0.0
        self.grad_update_steps = 0

        self.vocab_size = config.vocab_size
        self.trainable_params = [param for param in model.parameters() if param.requires_grad]
        self.optimizer, self.scheduler = setup_optimizer_and_scheduler(
            model, config.optimizer(), config.scheduler(), self.total_steps
        )

        self.max_grad_norm = config.clip_grad_norm

    def update_gradient(self) -> None:
        self.total_grad_norm = torch.nn.utils.clip_grad_norm_(self.trainable_params, self.max_grad_norm).item()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()

        self.grad_step = 0
        self.grad_update_steps += 1

    def train(self, model: TransformerModel, dataloader: DataLoader, config: TrainingConfig) -> None:
        """Main training loop for the Transformer language model."""
        if config.wandb_project:
            self.setup_wandb(config)

        self.setup_trainer(model, dataloader, config)

        model.train()
        self.optimizer.zero_grad(set_to_none=True)

        progress_bar: Any = tqdm(total=self.total_steps, desc="Training", unit="batch", dynamic_ncols=True)
        with progress_bar:
            for step, inputs in enumerate(dataloader):
                input_ids = inputs[0].to(device=DEVICE, non_blocking=True)
                x = input_ids[:, :-1]
                y = input_ids[:, 1:]

                # Forward pass with autocast enabled for mixed precision
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(x)
                    loss = compute_next_token_loss(logits, y)

                loss = loss / config.grad_accum_steps
                loss.backward()

                step_loss = loss.detach().item()
                if math.isnan(step_loss):
                    raise ValueError("NaN loss encountered")
                self.batch_loss += step_loss
                del logits, loss

                self.grad_step += 1
                if self.grad_step == config.grad_accum_steps:
                    self.update_gradient()

                    current_lr = self.optimizer.param_groups[0]["lr"]
                    progress_bar.update(1)
                    progress_bar.set_postfix(
                        {
                            "loss": f"{self.batch_loss:.4f}",
                            "lr": f"{current_lr:.2e}",
                        },
                        refresh=False,
                    )
                    progress_bar.refresh()
                    if self.wandb_run:
                        self.wandb_run.log(
                            data={
                                "train/loss": self.batch_loss,
                                "train/lr": current_lr,
                                "train/total_grad_norm": self.total_grad_norm,
                            },
                            step=step,
                        )

                    self.batch_loss = 0.0

        # Finish the wandb run
        if self.wandb_run:
            self.wandb_run.finish()

    def overfit(
        self, model: TransformerModel, dataloader: DataLoader, config: TrainingConfig, overfitting_epochs: int = 1000
    ) -> None:
        """Main training loop for the Transformer language model."""
        self.total_steps = overfitting_epochs
        self.setup_trainer(model, dataloader, config, overfit=True)

        model.train()
        self.optimizer.zero_grad(set_to_none=True)

        progress_bar: Any = tqdm(total=overfitting_epochs, desc="Training", unit="batch", dynamic_ncols=True)
        with progress_bar:
            for steps in range(overfitting_epochs):
                for _, inputs in enumerate(dataloader):
                    input_ids = inputs[0].to(device=DEVICE, non_blocking=True)
                    x = input_ids[:, :-1]
                    y = input_ids[:, 1:]

                    # Forward pass with autocast enabled for mixed precision
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits = model(x)
                        loss = compute_next_token_loss(logits, y)

                    loss = loss / config.grad_accum_steps
                    loss.backward()

                    step_loss = loss.detach().item()
                    if math.isnan(step_loss):
                        raise ValueError("NaN loss encountered")
                    self.batch_loss += step_loss
                    del logits, loss

                    self.grad_step += 1
                    if self.grad_step == config.grad_accum_steps:
                        if self.batch_loss < 1e-6:
                            sys.exit(f"Overfitted batch in: {steps} steps")

                        self.update_gradient()

                        current_lr = self.optimizer.param_groups[0]["lr"]
                        progress_bar.update(1)
                        progress_bar.set_postfix(
                            {
                                "loss": f"{self.batch_loss:.4f}",
                                "lr": f"{current_lr:.2e}",
                            },
                            refresh=False,
                        )
                        progress_bar.refresh()
                        self.batch_loss = 0.0


@click.command()
@click.option("--overfit", default=False, is_flag=True)
def main(*, overfit: bool) -> None:
    """Entry point for training the Transformer language model."""
    config = TrainingConfig(
        # wandb_project="gpt-2-speedrun",
        wandb_project=None
    )

    model = setup_llm(
        dimensions=config.llm_dimensions(),
        activation_function=config.activation_function,
        use_attn_bias=config.use_attn_bias,
        use_ffn_bias=config.use_ffn_bias,
        tie_encoder_decoder_weights=config.tie_encoder_decoder_weights,
        rms_norm=config.rms_norm,
        seq_len=config.seq_len,
        positional_encoding_type=config.positional_encoding_type,
        weight_init=config.weight_init(),
    )

    dataloader = prepare_data(
        tokenizer=model.tokenizer,
        batch_size=config.batch_size,
        grad_accum_steps=config.grad_accum_steps,
        load_single_batch=overfit,
    )

    trainer = Trainer()
    if overfit:
        trainer.overfit(model, dataloader, config)
    trainer.train(model, dataloader, config)


if __name__ == "__main__":
    main()
