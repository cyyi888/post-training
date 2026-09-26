"""训练策略注册表与工厂（策略模式入口）。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.registry import ALGORITHM_REGISTRY

if TYPE_CHECKING:
    from src.core.base import BaseTrainer


def get_trainer(algorithm: str) -> BaseTrainer:
    """
    按算法名取出训练策略实例。

    可用名：``sft`` / ``dpo`` / ``grpo`` / ``ppo``（以注册表为准）。
    """
    if algorithm not in ALGORITHM_REGISTRY:
        raise KeyError(
            f"未知训练策略 '{algorithm}'。已注册: {sorted(ALGORITHM_REGISTRY)}"
        )
    cls = ALGORITHM_REGISTRY[algorithm]
    return cls()


def available_algorithms() -> list[str]:
    return sorted(ALGORITHM_REGISTRY.keys())
