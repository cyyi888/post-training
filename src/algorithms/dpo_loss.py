"""Custom DPO / IPO loss with dynamic beta and optional penalty terms."""

from __future__ import annotations

from dataclasses import dataclass

try:
    from typing import Literal
except ImportError:  # Python < 3.8
    from typing_extensions import Literal  # type: ignore

import torch
import torch.nn.functional as F


LossType = Literal["dpo", "ipo"]
PenaltyType = Literal["length", "kl", "none"]


@dataclass
class DPOLossConfig:
    loss_type: LossType = "dpo"
    beta: float = 0.1
    dynamic_beta: bool = False
    beta_min: float = 0.05
    beta_max: float = 0.5
    label_smoothing: float = 0.0
    penalty_enabled: bool = False
    penalty_type: PenaltyType = "none"
    penalty_weight: float = 0.0


def _resolve_beta(cfg: DPOLossConfig, logits_diff: torch.Tensor) -> float | torch.Tensor:
    """Optionally scale beta by how separated chosen/rejected already are."""
    if not cfg.dynamic_beta:
        return cfg.beta
    # Larger |margin| → smaller beta (less aggressive update)
    margin = logits_diff.detach().abs().mean()
    scale = torch.sigmoid(-margin)  # ~0.5 when margin=0
    beta = cfg.beta_min + (cfg.beta_max - cfg.beta_min) * scale
    return beta


def _penalty(
    cfg: DPOLossConfig,
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    chosen_lengths: torch.Tensor | None = None,
    rejected_lengths: torch.Tensor | None = None,
    ref_chosen_logps: torch.Tensor | None = None,
) -> torch.Tensor:
    if not cfg.penalty_enabled or cfg.penalty_type == "none" or cfg.penalty_weight <= 0:
        return torch.zeros((), device=policy_chosen_logps.device)

    if cfg.penalty_type == "length":
        if chosen_lengths is None or rejected_lengths is None:
            return torch.zeros((), device=policy_chosen_logps.device)
        # Discourage overly long chosen completions relative to rejected
        gap = F.relu(chosen_lengths.float() - rejected_lengths.float()).mean()
        return cfg.penalty_weight * gap

    if cfg.penalty_type == "kl":
        if ref_chosen_logps is None:
            return torch.zeros((), device=policy_chosen_logps.device)
        # Approximate token-level KL via log-prob gap on chosen
        kl = (policy_chosen_logps - ref_chosen_logps).mean().abs()
        return cfg.penalty_weight * kl

    return torch.zeros((), device=policy_chosen_logps.device)


def compute_dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    cfg: DPOLossConfig | None = None,
    chosen_lengths: torch.Tensor | None = None,
    rejected_lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute DPO or IPO loss.

    Args:
        policy_*/ref_*: per-sequence log-probabilities, shape (B,).
        cfg: loss hyperparameters.
        chosen_lengths / rejected_lengths: optional token lengths for length penalty.

    Returns:
        (loss, metrics) where metrics contain reward margins and beta used.
    """
    cfg = cfg or DPOLossConfig()

    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = pi_logratios - ref_logratios  # implicit reward margin

    beta = _resolve_beta(cfg, logits)
    beta_val = float(beta) if isinstance(beta, (int, float)) else float(beta.mean().item())

    if cfg.loss_type == "ipo":
        # IPO: (logits - 1/(2*beta))^2
        losses = (logits - 1.0 / (2 * beta)) ** 2
    else:
        # Standard DPO with optional label smoothing
        losses = (
            -F.logsigmoid(beta * logits) * (1 - cfg.label_smoothing)
            - F.logsigmoid(-beta * logits) * cfg.label_smoothing
        )

    loss = losses.mean()
    pen = _penalty(
        cfg,
        policy_chosen_logps,
        policy_rejected_logps,
        chosen_lengths=chosen_lengths,
        rejected_lengths=rejected_lengths,
        ref_chosen_logps=ref_chosen_logps,
    )
    loss = loss + pen

    chosen_rewards = (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = (policy_rejected_logps - ref_rejected_logps).detach()
    metrics = {
        "loss": float(loss.detach().item()),
        "rewards/chosen": float(chosen_rewards.mean().item()),
        "rewards/rejected": float(rejected_rewards.mean().item()),
        "rewards/margins": float((chosen_rewards - rejected_rewards).mean().item()),
        "beta": beta_val,
        "penalty": float(pen.detach().item()) if pen.ndim == 0 else float(pen.mean().item()),
    }
    return loss, metrics


class DPOLoss(torch.nn.Module):
    """nn.Module wrapper around :func:`compute_dpo_loss`."""

    def __init__(self, cfg: DPOLossConfig | None = None):
        super().__init__()
        self.cfg = cfg or DPOLossConfig()

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
        chosen_lengths: torch.Tensor | None = None,
        rejected_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        return compute_dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            cfg=self.cfg,
            chosen_lengths=chosen_lengths,
            rejected_lengths=rejected_lengths,
        )
