"""动态 β 控制器：按验证 KL 相对目标自动加减惩罚强度。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class DynamicBetaState:
    beta: float
    init_beta: float
    target_kl: float
    adjust_rate: float
    min_beta: float
    max_beta: float
    last_kl: float | None = None
    steps: int = 0


class DynamicBetaController:
    """
    纯逻辑控制器，不依赖 PyTorch。

    - KL 高于目标 110% → 加大 β（更强约束）
    - KL 低于目标 90% → 减小 β（放松约束）
    - 中间带保持不变，并裁剪到 [min_beta, max_beta]
    """

    def __init__(
        self,
        init_beta: float = 0.1,
        target_kl: float = 4.0,
        adjust_rate: float = 0.1,
        min_beta: float = 0.01,
        max_beta: float = 1.0,
    ) -> None:
        self.init_beta = float(init_beta)
        self.target_kl = float(target_kl)
        self.adjust_rate = float(adjust_rate)
        self.min_beta = float(min_beta)
        self.max_beta = float(max_beta)
        self.beta = self._clip(self.init_beta)
        self.last_kl: float | None = None
        self.steps = 0

    def _clip(self, beta: float) -> float:
        return max(self.min_beta, min(self.max_beta, float(beta)))

    def step(self, current_kl: float) -> float:
        """根据当前验证集平均 KL 更新并返回新的 β。"""
        kl = float(current_kl)
        self.last_kl = kl
        high = self.target_kl * 1.1
        low = self.target_kl * 0.9
        if kl > high:
            # KL 过高：加强约束
            self.beta *= 1.0 + self.adjust_rate
        elif kl < low:
            # KL 过低：放松约束
            self.beta /= 1.0 + self.adjust_rate
        self.beta = self._clip(self.beta)
        self.steps += 1
        return self.beta

    def state_dict(self) -> dict[str, Any]:
        return asdict(
            DynamicBetaState(
                beta=self.beta,
                init_beta=self.init_beta,
                target_kl=self.target_kl,
                adjust_rate=self.adjust_rate,
                min_beta=self.min_beta,
                max_beta=self.max_beta,
                last_kl=self.last_kl,
                steps=self.steps,
            )
        )

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.beta = self._clip(float(state.get("beta", self.beta)))
        self.init_beta = float(state.get("init_beta", self.init_beta))
        self.target_kl = float(state.get("target_kl", self.target_kl))
        self.adjust_rate = float(state.get("adjust_rate", self.adjust_rate))
        self.min_beta = float(state.get("min_beta", self.min_beta))
        self.max_beta = float(state.get("max_beta", self.max_beta))
        last = state.get("last_kl", None)
        self.last_kl = None if last is None else float(last)
        self.steps = int(state.get("steps", 0))
