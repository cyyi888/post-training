"""Pluggable reward function engine for GRPO."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

RewardFn = Callable[..., list[float]]

_REGISTRY: dict[str, RewardFn] = {}


def register_reward(name: str):
    def deco(fn: RewardFn):
        _REGISTRY[name] = fn
        return fn

    return deco


def _extract_text(completion: Any) -> str:
    if isinstance(completion, list):
        return completion[0]["content"]
    return str(completion)


@register_reward("accuracy")
def reward_accuracy(completions: Sequence[Any], ground_truth: Sequence[str], **_: Any) -> list[float]:
    """Exact match on \\boxed{...} extraction vs ground truth."""
    scores = []
    for completion, gt in zip(completions, ground_truth):
        text = _extract_text(completion)
        match = re.search(r"\\boxed\{(.*?)\}", text)
        pred = match.group(1).strip() if match else ""
        scores.append(1.0 if pred == str(gt).strip() else 0.0)
    return scores


@register_reward("format")
def reward_format(completions: Sequence[Any], **_: Any) -> list[float]:
    """Reward presence of a \\boxed{...} block."""
    scores = []
    for completion in completions:
        text = _extract_text(completion)
        scores.append(1.0 if re.search(r"\\boxed\{.*?\}", text) else 0.0)
    return scores


class RewardEngine:
    """Combine multiple named reward functions with weights."""

    def __init__(self, names: Sequence[str], weights: Sequence[float] | None = None):
        missing = [n for n in names if n not in _REGISTRY]
        if missing:
            raise KeyError(f"Unknown reward funcs: {missing}. Available: {list(_REGISTRY)}")
        self.names = list(names)
        if weights is None:
            self.weights = [1.0] * len(self.names)
        else:
            if len(weights) != len(names):
                raise ValueError("weights length must match funcs")
            self.weights = list(weights)

    def __call__(self, completions: Sequence[Any], **kwargs: Any) -> list[float]:
        if not self.names:
            return [0.0] * len(completions)
        acc = [0.0] * len(completions)
        for name, w in zip(self.names, self.weights):
            partial = _REGISTRY[name](completions, **kwargs)
            for i, v in enumerate(partial):
                acc[i] += w * v
        return acc


def build_reward_funcs(cfg: Any) -> RewardFn:
    """Build a single callable from Hydra `training.reward` config."""
    names = list(cfg.funcs)
    weights = list(cfg.get("weights", [1.0] * len(names)))
    return RewardEngine(names, weights)
