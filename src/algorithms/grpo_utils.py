"""Helpers for GRPO group-relative advantage estimation."""

from __future__ import annotations

import torch


def group_relative_advantages(
    rewards: torch.Tensor,
    num_generations: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute group-relative advantages for GRPO.

    Args:
        rewards: shape (B * G,) flat rewards for G samples per prompt.
        num_generations: G.
        eps: numerical stability for std.

    Returns:
        Normalized advantages, same shape as rewards.
    """
    if rewards.ndim != 1:
        raise ValueError(f"Expected 1D rewards, got shape {tuple(rewards.shape)}")
    if rewards.numel() % num_generations != 0:
        raise ValueError(
            f"rewards length {rewards.numel()} not divisible by num_generations={num_generations}"
        )

    grouped = rewards.view(-1, num_generations)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True)
    adv = (grouped - mean) / (std + eps)
    return adv.view(-1)


def flatten_completions(completions: list, num_generations: int) -> list:
    """Ensure completions are a flat list of length B*G."""
    if len(completions) == 0:
        return completions
    # Already flat
    if not isinstance(completions[0], list) or (
        isinstance(completions[0], list)
        and completions[0]
        and isinstance(completions[0][0], dict)
    ):
        return completions
    flat = []
    for group in completions:
        flat.extend(group)
    if len(flat) % num_generations != 0:
        raise ValueError("flattened completions length mismatch with num_generations")
    return flat
