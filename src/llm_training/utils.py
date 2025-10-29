import torch
import torch.nn.functional as F


def compute_next_token_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # logits: [B, T, V], targets: [B, T]
    return F.cross_entropy(
        logits.movedim(-1, 1),  # -> [B, V, T]
        targets,
        ignore_index=-1,
        reduction="mean",
    )
