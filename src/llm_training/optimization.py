import math
from dataclasses import dataclass
from enum import StrEnum

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class OptimizerType(StrEnum):
    """Supported optimizer implementations."""

    SGD = "sgd"
    ADAMW = "adamw"


class DecaySchedule(StrEnum):
    """Supported learning rate scheduler types."""

    CONSTANT = "constant"
    COSINE = "cosine"
    LINEAR = "linear"


class WarmupType(StrEnum):
    """Learning rate warmup strategies."""

    NONE = "none"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    LOG = "log"
    COSINE = "cosine"


@dataclass
class OptimizerConfig:
    """Configuration parameters for the optimizer."""

    algorithm: OptimizerType  # Optimizer implementation to use
    lr: float  # Base learning rate for the optimizer
    weight_decay: float  # Weight decay strength applied by the optimizer
    eps: float  # Numerical stability term added to the denominator
    beta1: float
    beta2: float


@dataclass
class SchedulerConfig:
    """Configuration parameters for the learning rate scheduler."""

    # Warmup settings
    warmup_type: WarmupType  # Warmup function applied to the start of training
    warmup_ratio: float  # Fraction of training dedicated to warmup

    # Decay settings
    decay: DecaySchedule  # Learning rate scheduler decay type
    min_lr_ratio: float  # Relative learning rate maintained after decay completes.


def _wd_groups(
    model: nn.Module,
    *,
    wd: float,
    no_decay_names: tuple[str, ...] = (
        "bias",
        "LayerNorm.weight",
        "layer_norm.weight",
        "norm.weight",
        "BatchNorm.weight",
    ),
    exclude_dim_lt_2: bool = True,
) -> list[dict]:
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        by_name = any(n.endswith(s) for s in no_decay_names)
        by_dim = exclude_dim_lt_2 and (p.ndim < 2)
        (no_decay if (by_name or by_dim) else decay).append(p)
    assert decay, "no parameters left to decay"
    return [
        {"params": decay, "weight_decay": wd},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _create_optimizer(model: nn.Module, config: OptimizerConfig) -> Optimizer:
    if config.weight_decay == 0.0:
        groups = (p for p in model.parameters() if p.requires_grad)
    else:
        groups = _wd_groups(model, wd=config.weight_decay)

    match config.algorithm:
        case OptimizerType.ADAMW:
            return torch.optim.AdamW(
                params=groups,
                lr=config.lr,
                eps=config.eps,
                betas=(config.beta1, config.beta2),
                fused=True,
            )
        case OptimizerType.SGD:
            return torch.optim.SGD(
                params=groups,
                lr=config.lr,
            )


def _create_scheduler(
    optimizer: Optimizer,
    *,
    config: SchedulerConfig,
    total_steps: int,
) -> LRScheduler:
    warmup_steps = 0
    if config.warmup_type is not WarmupType.NONE:
        warmup_steps = max(1, int(total_steps * config.warmup_ratio))
    decay_span = max(1, total_steps - warmup_steps)

    def _warmup_factor(step: int) -> float:
        if warmup_steps == 0 or config.warmup_type is WarmupType.NONE:
            return 1.0
        if step >= warmup_steps:
            return 1.0
        progress = float(step + 1) / float(warmup_steps)
        match config.warmup_type:
            case WarmupType.LINEAR:
                return progress
            case WarmupType.EXPONENTIAL:
                return math.expm1(progress) / math.expm1(1.0)
            case WarmupType.LOG:
                return math.log1p(progress * (math.e - 1.0)) / math.log1p(math.e - 1.0)
            case WarmupType.COSINE:
                return 0.5 * (1.0 - math.cos(math.pi * progress))
        raise ValueError(f"Unsupported warmup type: {config.warmup_type}.")

    def _decay_factor(step: int) -> float:
        if config.decay is DecaySchedule.CONSTANT:
            return 1.0
        progress = (step - warmup_steps + 1) / float(decay_span)
        progress = min(max(progress, 0.0), 1.0)
        match config.decay:
            case DecaySchedule.COSINE:
                base_multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
            case DecaySchedule.LINEAR:
                base_multiplier = max(0.0, 1.0 - progress)
            case _:
                raise ValueError(f"Unsupported decay schedule: {config.decay}.")
        return config.min_lr_ratio + (1.0 - config.min_lr_ratio) * base_multiplier

    def lr_lambda(current_step: int) -> float:
        warmup = _warmup_factor(current_step)
        decay = _decay_factor(current_step)
        return warmup * decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def setup_optimizer_and_scheduler(
    model: nn.Module, optimizer_config: OptimizerConfig, scheduler_config: SchedulerConfig, total_steps: int
) -> tuple[Optimizer, LRScheduler]:
    """Setup the training optimizer and learning-rate scheduler."""
    optimizer = _create_optimizer(model, optimizer_config)
    scheduler = _create_scheduler(optimizer, config=scheduler_config, total_steps=total_steps)
    return optimizer, scheduler
