"""PostTrainLab core interfaces and base classes."""

from .base import BaseEvaluator, BaseTrainer, RuntimeOptions, TrainContext, TrainResult
from .registry import ALGORITHM_REGISTRY, DATA_REGISTRY, register

__all__ = [
    "BaseTrainer",
    "BaseEvaluator",
    "RuntimeOptions",
    "TrainContext",
    "TrainResult",
    "ALGORITHM_REGISTRY",
    "DATA_REGISTRY",
    "register",
]
