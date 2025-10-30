from enum import StrEnum

import torch
import torch.nn.functional as F
from torch import nn


class ActivationFunction(StrEnum):
    """Activation function to use in fully connected blocks."""

    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    RELU2 = "relu^2"
    GLU = "glu"
    GELU = "gelu"
    SWISH = "swish"
    SWIGLU = "swiglu"

    def module(self) -> nn.Module:
        match self:
            case ActivationFunction.RELU:
                module = nn.ReLU()
            case ActivationFunction.LEAKY_RELU:
                module = nn.LeakyReLU()
            case ActivationFunction.RELU2:
                module = ReLU2()
            case ActivationFunction.GLU:
                module = nn.GLU()
            case ActivationFunction.GELU:
                module = nn.GELU()
            case ActivationFunction.SWISH:
                module = Swish()
            case ActivationFunction.SWIGLU:
                module = SwiGLU()
            case _:
                raise ValueError(f"Unknown activation function: {self}")
        return module

    @property
    def width_multiplier(self) -> int:
        """Multiplier applied to the feed-forward hidden width before activation."""
        match self:
            case ActivationFunction.GLU | ActivationFunction.SWIGLU:
                return 2
            case _:
                return 1


class ReLU2(nn.Module):
    """ReLU Squared activation function."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x).square()


class Swish(nn.Module):
    """Swish activation function."""

    def __init__(self) -> None:
        super().__init__()
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * F.sigmoid(self.beta * x)


class SwiGLU(nn.Module):
    """SwiGLU activation function."""

    def __init__(self) -> None:
        super().__init__()
        self.swish = Swish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = x.chunk(2, dim=-1)
        return value * self.swish(gate)
