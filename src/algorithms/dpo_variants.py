"""DPO 变体分发：IPO 等。"""

from __future__ import annotations

from typing import Any

from src.algorithms.dpo_loss import DPOLossConfig
from src.algorithms.dpo_losses import compute_preference_loss, ipo_loss as _ipo_loss_fn


def compute_ipo_loss(
    policy_chosen_logps: Any,
    policy_rejected_logps: Any,
    ref_chosen_logps: Any,
    ref_rejected_logps: Any,
    cfg: DPOLossConfig | None = None,
    beta: float | None = None,
    margin: float | None = None,
    **_kwargs: Any,
) -> tuple[Any, dict[str, float]]:
    """IPO 合页变体。"""
    cfg = cfg or DPOLossConfig(loss_type="ipo")
    use_beta = float(cfg.beta if beta is None else beta)
    use_margin = float(cfg.margin if margin is None else margin)
    return _ipo_loss_fn(
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        beta=use_beta,
        margin=use_margin,
    )


def compute_variant_loss(
    loss_type: str,
    policy_chosen_logps: Any,
    policy_rejected_logps: Any,
    ref_chosen_logps: Any,
    ref_rejected_logps: Any,
    cfg: DPOLossConfig | None = None,
    **kwargs: Any,
) -> tuple[Any, dict[str, float]]:
    """按 loss_type 分发到 DPO / IPO。"""
    cfg = cfg or DPOLossConfig()
    beta = float(kwargs.pop("beta", cfg.beta))
    margin = float(kwargs.pop("margin", cfg.margin))
    return compute_preference_loss(
        loss_type or cfg.loss_type,
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        beta=beta,
        margin=margin,
    )
