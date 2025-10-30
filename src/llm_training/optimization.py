import math
from dataclasses import dataclass
from enum import StrEnum

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

WEIGHT_DECAY_EXCLUSIONS = ["embeddings", "position", "bias", "norm", "lm_head"]


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
    weight_decay: float,
    min_dim_for_decay: int = 2,
) -> list[dict]:
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        by_name = any(s in n for s in WEIGHT_DECAY_EXCLUSIONS)
        by_dim = p.ndim < min_dim_for_decay
        if by_name or by_dim:
            no_decay.append(p)
        else:
            decay.append(p)

    assert decay, "no parameters left to decay"
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def create_optimizer(model: nn.Module, config: OptimizerConfig) -> Optimizer:
    """Construct an optimizer over model parameters with optional weight decay groups."""

    trainable_params = _wd_groups(model, weight_decay=config.weight_decay)

    match config.algorithm:
        case OptimizerType.ADAMW:
            return torch.optim.AdamW(
                params=trainable_params,
                lr=config.lr,
                eps=config.eps,
                betas=(config.beta1, config.beta2),
                fused=True,
            )
        case OptimizerType.SGD:
            return torch.optim.SGD(
                params=trainable_params,
                lr=config.lr,
            )
        case _:
            raise ValueError(f"Unsupported optimizer type: {config.algorithm}")


def create_scheduler(
    optimizer: Optimizer,
    *,
    config: SchedulerConfig,
    total_steps: int,
) -> LRScheduler:
    """Create a learning rate scheduler with warmup and decay phases."""
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

    def _lr_lambda(current_step: int) -> float:
        warmup = _warmup_factor(current_step)
        decay = _decay_factor(current_step)
        return warmup * decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
