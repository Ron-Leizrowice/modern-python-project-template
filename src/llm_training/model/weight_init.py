import math
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

import torch
from torch import nn

from llm_training.model.blocks import TransformerBlock
from llm_training.model.pe import LearnedPositionalEncoding


class WeightInitType(StrEnum):
    """Supported weight initialization strategies."""

    UNIFORM = "uniform"
    NORMAL = "normal"
    SPECTRAL = "spectral"


@dataclass
class WeightInitConfig:
    """Config class to control weight initialization for a specific component."""

    dtype: torch.dtype
    type: WeightInitType
    std: float
    range: float = 0.0


@dataclass
class ModelWeightInit:
    """Config class to determine weight init for all model components."""

    embedding: WeightInitConfig  # also used for lm-head
    pe: WeightInitConfig
    ff: WeightInitConfig
    attn: WeightInitConfig


def _init_tensor(tensor: torch.Tensor, config: WeightInitConfig) -> None:
    match config.type:
        case WeightInitType.UNIFORM:
            nn.init.uniform_(tensor=tensor, a=-config.range / 2, b=config.range / 2)
        case WeightInitType.NORMAL:
            nn.init.normal_(tensor=tensor, mean=0.0, std=config.std)
        case WeightInitType.SPECTRAL:
            fan_out = tensor.size(0)
            fan_in = tensor.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(tensor, mean=0.0, std=std)

    tensor.to(config.dtype)


def init_embedding(module: nn.Embedding | nn.Linear, config: WeightInitConfig) -> None:
    _init_tensor(module.weight, config)


def init_positional_encoding(module: nn.Module, config: WeightInitConfig) -> None:
    if isinstance(module, LearnedPositionalEncoding):
        _init_tensor(module.position_embeddings.weight, config)


def _init_linear_layer(module: nn.Linear, config: WeightInitConfig) -> None:
    _init_tensor(module.weight, config=config)

    if module.bias is not None:
        torch.nn.init.zeros_(module.bias)


def init_transformer_block(module: nn.Module, ff_config: WeightInitConfig, attn_config: WeightInitConfig) -> None:
    block = cast("TransformerBlock", module)

    attn = block.attn
    _init_linear_layer(attn.q_proj, attn_config)
    _init_linear_layer(attn.k_proj, attn_config)
    _init_linear_layer(attn.v_proj, attn_config)
    _init_linear_layer(attn.out_proj, attn_config)

    ff = block.ff
    _init_linear_layer(ff.fc, ff_config)
    _init_linear_layer(ff.proj, ff_config)
