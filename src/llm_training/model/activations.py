from enum import StrEnum

import torch
import torch.nn.functional as F


class ActivationFunction(StrEnum):
    """Activation function to use in fully connected blocks."""

    RELU = "relu"
    RELU2 = "relu^2"
    GELU = "gelu"


def _relu_squared(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x).square()


def _relu(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x)


def _gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x)
