"""兼容再导出：变体入口已并入 ``dpo_losses``。"""

from __future__ import annotations

from src.algorithms.dpo_losses import (
    DPOLossConfig,
    compute_ipo_loss,
    compute_preference_loss,
    compute_variant_loss,
)

__all__ = [
    "DPOLossConfig",
    "compute_ipo_loss",
    "compute_preference_loss",
    "compute_variant_loss",
]
