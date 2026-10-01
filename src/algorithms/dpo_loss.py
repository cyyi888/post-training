"""DPO Loss 配置与兼容入口。

底层公式见 ``dpo_losses``；变体分发见 ``dpo_variants``。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.algorithms.dpo_losses import compute_preference_loss, dpo_loss as _dpo_loss_fn

LossType = str  # "dpo" | "ipo"
PenaltyType = str  # "length" | "kl" | "none"
BetaMode = str  # "fixed" | "dynamic"


@dataclass
class DPOLossConfig:
    """DPO / IPO 超参。"""

    loss_type: LossType = "dpo"
    beta: float = 0.1
    beta_mode: BetaMode = "fixed"
    dynamic_beta: bool = False  # 兼容旧配置：True ↔ beta_mode=dynamic
    beta_min: float = 0.01
    beta_max: float = 1.0
    target_kl: float = 4.0
    adjust_rate: float = 0.1
    margin: float = 1.0
    label_smoothing: float = 0.0
    penalty_enabled: bool = False
    penalty_type: PenaltyType = "none"
    penalty_weight: float = 0.0
    reference_free: bool = False

    def resolved_beta_mode(self) -> str:
        if self.beta_mode in ("fixed", "dynamic"):
            if self.dynamic_beta and self.beta_mode == "fixed":
                return "dynamic"
            return str(self.beta_mode)
        return "dynamic" if self.dynamic_beta else "fixed"


def compute_dpo_loss(
    policy_chosen_logps: Any,
    policy_rejected_logps: Any,
    ref_chosen_logps: Any,
    ref_rejected_logps: Any,
    cfg: DPOLossConfig | None = None,
    chosen_lengths: Any = None,
    rejected_lengths: Any = None,
    beta: float | None = None,
    **_kwargs: Any,
) -> tuple[Any, dict[str, float]]:
    """计算标准 DPO 损失（忽略长度惩罚占位参数）。"""
    del chosen_lengths, rejected_lengths
    cfg = cfg or DPOLossConfig()
    use_beta = float(cfg.beta if beta is None else beta)
    return _dpo_loss_fn(
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        beta=use_beta,
    )


class DPOLoss:
    """Loss 模块包装；真正计算仍走 :func:`compute_dpo_loss` / preference loss。"""

    def __init__(self, cfg: DPOLossConfig | None = None):
        self.cfg = cfg or DPOLossConfig()

    def __call__(self, *args: Any, **kwargs: Any):
        kind = str(kwargs.pop("loss_type", self.cfg.loss_type))
        beta = float(kwargs.pop("beta", self.cfg.beta))
        margin = float(kwargs.pop("margin", self.cfg.margin))
        if len(args) >= 4:
            return compute_preference_loss(
                kind,
                args[0],
                args[1],
                args[2],
                args[3],
                beta=beta,
                margin=margin,
            )
        return compute_dpo_loss(*args, cfg=self.cfg, beta=beta, **kwargs)

    def forward(self, *args: Any, **kwargs: Any):
        return self.__call__(*args, **kwargs)
