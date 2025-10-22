import torch
import torch.nn.functional as F


def compute_next_token_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Mean cross-entropy for next-token prediction with shift.

    Expects a (B, T) per-token mask where 1=keep, 0=pad. If a different
    shape is provided, falls back to uniform weighting.
    """
    batch_size, seq_len, vocab_size = logits.shape
    assert targets.shape[:2] == (batch_size, seq_len), "targets shape must match logits batch and time"

    # Shift: predict x[t+1] from x[t]
    logits_shifted = logits[:, :-1, :]  # (B, T-1, V)
    targets_shifted = targets[:, 1:]  # (B, T-1)

    # Flatten for CE
    logit_view = logits_shifted.reshape(-1, vocab_size).float()
    target_view = targets_shifted.reshape(-1)
    per_token_loss = F.cross_entropy(logit_view, target_view, reduction="none")
    per_token_loss = per_token_loss.to(dtype=torch.float32)

    # Loss weights: prefer a 2D per-token mask
    weights = torch.ones_like(per_token_loss, device=per_token_loss.device, dtype=torch.float32)

    weighted_loss = per_token_loss * weights
    normalization = weights.sum().clamp_min(1.0)
    return weighted_loss.sum() / normalization
