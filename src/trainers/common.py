"""HF/TRL 日志参数。新代码请用 ``BaseTrainer.hf_training_kwargs``。"""

from __future__ import annotations

from typing import Any

from src.core.base import BaseTrainer


def trainer_logging_kwargs(cfg: Any) -> dict[str, Any]:
    """兼容旧调用：不含分布式 rank（那部分在 BaseTrainer.train 里组装）。"""
    helper = BaseTrainer()
    return helper.hf_training_kwargs(cfg)
