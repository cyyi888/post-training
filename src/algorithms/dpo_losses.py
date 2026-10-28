"""自研 DPO / IPO 损失：公式实现 + 配置。

公式对应：
- DPO: L = -E[ log σ(β · (log πθ(y+|x)/πref(y+|x) - log πθ(y-|x)/πref(y-|x))) ]
- IPO(合页): L = E[ max(0, margin - β · (r+ - r-)) ]

公开 API：
- 配置：``DPOLossConfig``
- 公式：``dpo_loss`` / ``ipo_loss``
- 分发：``compute_preference_loss``
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
    """DPO / IPO 超参。

    β 口径：``beta_mode``（fixed|dynamic）+ 运行时标量 ``beta``。
    Hydra 侧用 ``training.fixed_beta`` / ``training.dynamic_beta.*``，
    由 trainer 解析后写入本配置。
    """

    loss_type: LossType = "dpo"
    beta: float = 0.1  # 当前使用的 β（fixed 时=fixed_beta；dynamic 时=init/控制器值）
    beta_mode: BetaMode = "fixed"
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
        mode = str(self.beta_mode or "fixed").lower()
        if mode not in ("fixed", "dynamic"):
            raise ValueError(f"beta_mode 必须是 fixed|dynamic，收到: {self.beta_mode!r}")
        return mode


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
