from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainResult:
    """Normalized training outcome shared by all trainers."""

    metrics: dict[str, float] = field(default_factory=dict)
    checkpoint_dir: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class BaseTrainer(ABC):
    """Unified trainer interface for SFT / DPO / GRPO stages."""

    def __init__(self, cfg: Any, model: Any, tokenizer: Any, dataset: Any):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset

    @abstractmethod
    def train(self) -> TrainResult:
        """Run training and return aggregated metrics."""

    def save(self, path: str) -> None:
        """Persist model + tokenizer; override when needed."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


class BaseEvaluator(ABC):
    """Unified evaluation interface."""

    def __init__(self, cfg: Any, model: Any, tokenizer: Any):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer

    @abstractmethod
    def evaluate(self, dataset: Any) -> dict[str, float]:
        """Return metric name → value."""
