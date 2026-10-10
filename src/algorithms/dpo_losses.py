"""自研 DPO / IPO 损失：公式实现 + 配置 + 兼容入口。

公式对应：
- DPO: L = -E[ log σ(β · (log πθ(y+|x)/πref(y+|x) - log πθ(y-|x)/πref(y-|x))) ]
- IPO(合页): L = E[ max(0, margin - β · (r+ - r-)) ]

公开 API：
- 底层：``dpo_loss`` / ``ipo_loss`` / ``compute_preference_loss``
- 配置：``DPOLossConfig``
- 兼容：``compute_dpo_loss`` / ``compute_ipo_loss`` / ``compute_variant_loss`` / ``DPOLoss``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

LossType = str  # "dpo" | "ipo"
PenaltyType = str  # "length" | "kl" | "none"
BetaMode = str  # "fixed" | "dynamic"


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 底层公式
# ---------------------------------------------------------------------------


def _as_float_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.as_tensor(x, dtype=torch.float32)


def _preference_terms(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回 (r+, r-, delta=r+-r-)。"""
    r_pos = policy_chosen_logps - ref_chosen_logps
    r_neg = policy_rejected_logps - ref_rejected_logps
    delta = r_pos - r_neg
    return r_pos, r_neg, delta


@torch.no_grad()
def _build_metrics(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    delta: torch.Tensor,
    beta: float,
) -> dict[str, float]:
    """监控：平均 KL、正负例对数比差。"""
    mean_kl = 0.5 * (
        (policy_chosen_logps - ref_chosen_logps).abs().mean()
        + (policy_rejected_logps - ref_rejected_logps).abs().mean()
    )
    logps_diff = delta.mean()
    return {
        "mean_kl": float(mean_kl.item()),
        "logps_diff": float(logps_diff.item()),
        "beta": float(beta),
        "rewards/margins": float((beta * delta).mean().item()),
    }


def dpo_loss(
    policy_chosen_logps: Any,
    policy_rejected_logps: Any,
    ref_chosen_logps: Any,
    ref_rejected_logps: Any,
    beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    标准 DPO Loss。

    使用 ``F.logsigmoid`` 保证数值稳定；禁止 ``log(sigmoid(...))``。
    """
    pc = _as_float_tensor(policy_chosen_logps)
    pr = _as_float_tensor(policy_rejected_logps)
    rc = _as_float_tensor(ref_chosen_logps)
    rr = _as_float_tensor(ref_rejected_logps)

    _r_pos, _r_neg, delta = _preference_terms(pc, pr, rc, rr)
    loss = -F.logsigmoid(float(beta) * delta).mean()
    metrics = _build_metrics(pc, pr, rc, rr, delta, float(beta))
    metrics["loss"] = float(loss.detach().item())
    return loss, metrics


def ipo_loss(
    policy_chosen_logps: Any,
    policy_rejected_logps: Any,
    ref_chosen_logps: Any,
    ref_rejected_logps: Any,
    beta: float,
    margin: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    IPO 合页变体：强制 β·(r+ - r-) 至少达到 margin。

    L = mean(max(0, margin - β · (r+ - r-)))
    """
    pc = _as_float_tensor(policy_chosen_logps)
    pr = _as_float_tensor(policy_rejected_logps)
    rc = _as_float_tensor(ref_chosen_logps)
    rr = _as_float_tensor(ref_rejected_logps)

    _r_pos, _r_neg, delta = _preference_terms(pc, pr, rc, rr)
    loss = F.relu(float(margin) - float(beta) * delta).mean()
    metrics = _build_metrics(pc, pr, rc, rr, delta, float(beta))
    metrics["loss"] = float(loss.detach().item())
    metrics["margin"] = float(margin)
    return loss, metrics


def compute_preference_loss(
    loss_type: str,
    policy_chosen_logps: Any,
    policy_rejected_logps: Any,
    ref_chosen_logps: Any,
    ref_rejected_logps: Any,
    *,
    beta: float,
    margin: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """按 loss_type 分发到 DPO / IPO。"""
    kind = str(loss_type or "dpo").lower()
    if kind in ("dpo", "sigmoid"):
        return dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            beta=beta,
        )
    if kind == "ipo":
        return ipo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            beta=beta,
            margin=margin,
        )
    raise ValueError(f"未知 loss_type={loss_type!r}，可选: dpo | ipo")


# ---------------------------------------------------------------------------
# 兼容入口（Config / 旧调用名）
# ---------------------------------------------------------------------------


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
    return dpo_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        beta=use_beta,
    )


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
    """IPO 合页变体（Config 兼容入口）。"""
    cfg = cfg or DPOLossConfig(loss_type="ipo")
    use_beta = float(cfg.beta if beta is None else beta)
    use_margin = float(cfg.margin if margin is None else margin)
    return ipo_loss(
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
    """按 loss_type 分发到 DPO / IPO（Config 兼容入口）。"""
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


class DPOLoss:
    """Loss 模块包装；真正计算走 preference / DPO 入口。"""

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
