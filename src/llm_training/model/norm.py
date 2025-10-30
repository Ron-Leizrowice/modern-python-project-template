from enum import StrEnum

import torch
import torch.nn.functional as F
from torch import nn


class NormFunction(StrEnum):
    """Norm functions."""

    LayerNorm = "layer_norm"
    FunctionalRmsNorm = "functional_rms_norm"


class FunctionalRmsNorm(nn.Module):
    """Purely functional rmsnorm with no learnable params."""

    __slots__ = ("dim",)

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (self.dim,))
