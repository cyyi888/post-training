"""任务评测器工厂。"""

from __future__ import annotations

from typing import Any

from src.core.base import BaseEvaluator
from src.core.registry import EVALUATOR_REGISTRY


def _ensure_registered() -> None:
    # 侧载注册，避免只 import factory 时注册表为空
    import src.evaluators.gsm8k_eval  # noqa: F401
    import src.evaluators.humaneval_eval  # noqa: F401


def build_evaluator(task: str, cfg: Any, model: Any, tokenizer: Any) -> BaseEvaluator:
    """按任务名构造评测器（``gsm8k`` / ``humaneval``）。"""
    _ensure_registered()
    key = str(task).lower().strip()
    # 兼容 data.name 别名
    aliases = {
        "openai_humaneval": "humaneval",
        "human_eval": "humaneval",
        "gsm-8k": "gsm8k",
    }
    key = aliases.get(key, key)
    if key not in EVALUATOR_REGISTRY:
        raise KeyError(
            f"未知评测任务 '{task}'。已注册: {sorted(EVALUATOR_REGISTRY)}"
        )
    return EVALUATOR_REGISTRY[key](cfg, model, tokenizer)


def available_tasks() -> list[str]:
    _ensure_registered()
    return sorted(EVALUATOR_REGISTRY.keys())
