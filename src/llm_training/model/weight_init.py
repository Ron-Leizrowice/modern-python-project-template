"""Tensor dimension notation.

- Fo: fan-out dimension (rows of weight matrices)
- Fi: fan-in dimension (columns of weight matrices)
- E: embedding width
- V: vocabulary size
- T: sequence length (positions)
"""

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, cast

import torch
from torch import nn

from llm_training.model.pe import LearnedPositionalEncoding

if TYPE_CHECKING:
    from llm_training.model.transformer_blocks import TransformerBlock


class WeightDistribution(StrEnum):
    """Supported weight initialization strategies."""

    UNIFORM = "uniform"
    NORMAL = "normal"
    SPECTRAL = "spectral"


@dataclass
class WeightInitConfig:
    """Config class to control weight initialization for a specific component."""

    dtype: torch.dtype
    dist: WeightDistribution
    std: float
    range: float = 0.0


@dataclass
class AttnWeightInitConfig(WeightInitConfig):
    """Config class to control weight initialization for attention layers."""

    proj_scale_factor: float = 1.0


@dataclass
class ModelWeightInit:
    """Config class to determine weight init for all model components."""

    embedding: WeightInitConfig  # also used for lm-head
    pe: WeightInitConfig
    ff: WeightInitConfig
    attn: AttnWeightInitConfig
    tie_encoder_decoder_weights: bool


def _init_tensor(
    tensor: torch.Tensor,
    dist: WeightDistribution,
    dist_range: float = 0.0,
    std_dev: float = 0.0,
    dtype: torch.dtype = torch.float32,
) -> None:
    match dist:
        case WeightDistribution.UNIFORM:
            nn.init.uniform_(tensor=tensor, a=-dist_range / 2, b=dist_range / 2)
        case WeightDistribution.NORMAL:
            nn.init.normal_(tensor=tensor, mean=0.0, std=std_dev)
        case WeightDistribution.SPECTRAL:
            fan_out = tensor.size(0)  # Fo
            fan_in = tensor.size(1)  # Fi
            std_dev = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(tensor, mean=0.0, std=std_dev)

    tensor.to(dtype)


def init_embedding(module: nn.Embedding | nn.Linear, config: WeightInitConfig) -> None:
    """Initialize embedding or LM-head matrices shaped (V, E) or (E, V)."""
    _init_tensor(
        module.weight, dist=config.dist, dist_range=config.range, std_dev=config.std, dtype=config.dtype
    )  # (V, E) for embeddings / (E, V) for lm head


def init_positional_encoding(module: nn.Module, config: WeightInitConfig) -> None:
    """Initialize positional embedding matrices shaped (T, E) when learnable."""
    if isinstance(module, LearnedPositionalEncoding):
        _init_tensor(
            module.position_embeddings.weight,
            dist=config.dist,
            dist_range=config.range,
            std_dev=config.std,
            dtype=config.dtype,
        )  # (T, E)


def _init_linear_layer(module: nn.Linear, config: WeightInitConfig, std_scale_factor: float = 1.0) -> None:
    std_dev = config.std / std_scale_factor
    _init_tensor(module.weight, dist=config.dist, dist_range=config.range, std_dev=std_dev, dtype=config.dtype)

    if module.bias is not None:
        torch.nn.init.zeros_(module.bias)


def _init_norm(module: nn.Module) -> None:
    if isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def init_transformer_block(module: nn.Module, ff_config: WeightInitConfig, attn_config: AttnWeightInitConfig) -> None:
    """Initialize attention and feed-forward projections within a Transformer block."""
    block = cast("TransformerBlock", module)

    attn = block.attn
    _init_linear_layer(attn.w_q, attn_config)
    _init_linear_layer(attn.w_k, attn_config)
    _init_linear_layer(attn.w_v, attn_config)
    _init_linear_layer(attn.w_proj, attn_config, std_scale_factor=attn_config.proj_scale_factor)

    ff = block.ff
    _init_linear_layer(ff.fc, ff_config)
    _init_linear_layer(ff.proj, ff_config)

    _init_norm(block.norm1)
    _init_norm(block.norm2)
