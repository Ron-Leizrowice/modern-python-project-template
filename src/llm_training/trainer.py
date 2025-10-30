"""Core training loop.

The goal of this project is to optimize LLM pre-training performance of a GPT-2 like model.
"""

import math
import sys
import time
from dataclasses import asdict

import torch
import wandb
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm.auto import tqdm

from llm_training.config import TrainingConfig
from llm_training.constants import DEVICE, WANDB_LOGGING_DIR
from llm_training.data import create_dataloader
from llm_training.model.gpt import (
    TransformerModel,
)
from llm_training.optimization import (
    create_optimizer,
    create_scheduler,
)
from llm_training.tokenizer import get_tokenizer
from llm_training.utils import compute_next_token_loss

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision("high")
torch.backends.cuda.enable_flash_sdp(enabled=True)
torch.backends.cuda.enable_mem_efficient_sdp(enabled=True)
torch.backends.cuda.enable_math_sdp(enabled=False)


class Trainer:
    """Helper class to manage training state."""

    config: TrainingConfig

    grad_step: int
    total_grad_norm: float
    grad_update_steps: int

    batch_loss: float

    scheduler: LRScheduler
    optimizer: Optimizer
    max_grad_norm: float

    wandb_run: wandb.Run | None = None
    progress_bar: tqdm

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def setup_wandb(self) -> None:
        self.wandb_run = wandb.init(
            project=self.config.wandb_project, config=asdict(self.config), dir=WANDB_LOGGING_DIR
        )

    def setup_model(self, *, compile_mode: str | None = "default") -> None:
        """Instantiate and prepare the Transformer model for training."""
        self.tokenizer = get_tokenizer(seq_len=self.config.seq_len, vocab_size=self.config.vocab_size)
        model = TransformerModel(self.config.llm_architecture())

        model = model.to(device=DEVICE)
        if compile_mode:
            model.compile(fullgraph=True, dynamic=False, mode=compile_mode)
        self.model = model

    def setup_dataloader(self, *, overfit: bool = False) -> None:
        self.dataloader = create_dataloader(
            tokenizer=self.tokenizer,
            batch_size=self.config.batch_size,
            grad_accum_steps=self.config.grad_accum_steps,
            overfit_single_batch=overfit,
        )

    def setup_optimization(self) -> None:
        self.trainable_params = [param for param in self.model.parameters() if param.requires_grad]
        self.optimizer = create_optimizer(self.model, self.config.optimizer())
        self.scheduler = create_scheduler(self.optimizer, config=self.config.scheduler(), total_steps=self.total_steps)

    def setup_training(self, *, max_steps: int | None = None) -> None:
        assert len(self.dataloader) % self.config.grad_accum_steps == 0

        # If a maximum number of steps is not specified, calculate from the total data
        if not max_steps:
            self.total_steps = len(self.dataloader) // self.config.grad_accum_steps
        else:
            self.total_steps = max_steps

        self.grad_step = 0
        self.total_grad_norm = 0.0
        self.batch_loss = 0.0
        self.grad_update_steps = 0

        self.setup_optimization()

    def update_gradient(self) -> None:
        self.total_grad_norm = torch.nn.utils.clip_grad_norm_(self.trainable_params, self.config.clip_grad_norm).item()

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()

        self.grad_step = 0
        self.grad_update_steps += 1
        self.batch_loss = 0.0

    def _process_step(self, inputs: torch.Tensor) -> None:
        input_ids = inputs[0].to(device=DEVICE, non_blocking=True)
        x = input_ids[:, :-1]
        y = input_ids[:, 1:]

        # Forward pass with autocast enabled for mixed precision
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = self.model(x)
            loss = compute_next_token_loss(logits, y)

        loss = loss / self.config.grad_accum_steps
        loss.backward()

        step_loss = loss.detach().item()
        if math.isnan(step_loss):
            raise ValueError("NaN loss encountered")
        self.batch_loss += step_loss
        del logits, loss

        self.grad_step += 1

    def _log_step(self, step: int) -> None:
        current_lr = self.optimizer.param_groups[0]["lr"]
        self.progress_bar.update(1)
        self.progress_bar.set_postfix(
            {
                "loss": f"{self.batch_loss:.4f}",
                "lr": f"{current_lr:.2e}",
            },
            refresh=False,
        )
        self.progress_bar.refresh()
        if self.wandb_run:
            self.wandb_run.log(
                data={
                    "train/loss": self.batch_loss,
                    "train/lr": current_lr,
                    "train/total_grad_norm": self.total_grad_norm,
                },
                step=step,
            )

    def train(self) -> None:
        """Main training loop for the Transformer language model."""
        if self.config.wandb_project:
            self.setup_wandb()

        self.setup_training()

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        self.progress_bar = tqdm(total=self.total_steps, desc="Training", unit="batch", dynamic_ncols=True)
        with self.progress_bar:
            for step, inputs in enumerate(self.dataloader):
                self._process_step(inputs)

                if self.grad_step == self.config.grad_accum_steps:
                    self._log_step(step)
                    self.update_gradient()

        # Finish the wandb run
        if self.wandb_run:
            self.wandb_run.finish()

    def overfit(
        self, overfitting_epochs: int, overfit_threshold: float = 1e-4, timing_warmup_ratio: float = 0.1
    ) -> None:
        """Main training loop for the Transformer language model."""
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        self.setup_training(max_steps=overfitting_epochs)

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        epoch_times: list[float] = []

        self.progress_bar = tqdm(total=overfitting_epochs, desc="Overfitting", unit="epoch", dynamic_ncols=True)
        with self.progress_bar:
            for epoch in range(overfitting_epochs):
                t0 = time.perf_counter()
                for step, inputs in enumerate(self.dataloader):
                    self._process_step(inputs)

                    if self.grad_step == self.config.grad_accum_steps:
                        self._log_step(step)
                        if self.batch_loss < overfit_threshold:
                            avg_epoch_time = sum(epoch_times) / len(epoch_times)
                            sys.exit(f"Overfitted in: {epoch + 1} epochs, avg epoch time: {avg_epoch_time:.2}s.")
                        self.update_gradient()
                if epoch >= int(overfitting_epochs * timing_warmup_ratio):
                    epoch_times.append(time.perf_counter() - t0)

        avg_epoch_time = round(sum(epoch_times) / len(epoch_times), 3)
        max_mem_allocated = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
        sys.exit(f"Failed to overfit batch, avg epoch time: {avg_epoch_time}s, max vram: {max_mem_allocated}")
