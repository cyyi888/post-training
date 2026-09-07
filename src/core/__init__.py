"""PostTrainLab core interfaces and base classes."""

from .base import BaseEvaluator, BaseTrainer, TrainResult
from .registry import ALGORITHM_REGISTRY, DATA_REGISTRY, register

__all__ = [
    "BaseTrainer",
    "BaseEvaluator",
    "TrainResult",
    "ALGORITHM_REGISTRY",
    "DATA_REGISTRY",
    "register",
]
