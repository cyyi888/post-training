"""SFT / 通用训练的梯度裁剪策略。

策略：
- ``norm``：全局 L2 范数裁剪（HF 默认行为）
- ``value``：逐元素绝对值裁剪
- ``adaptive``：用历史梯度范数的分位数 / EMA 动态调整阈值
- ``none``：不裁剪
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable

import torch
from torch import nn


@dataclass
class GradClipConfig:
    strategy: str = "norm"  # norm | value | adaptive | none
    max_norm: float = 1.0
    max_value: float = 1.0
    adaptive_window: int = 100
    adaptive_percentile: float = 90.0
    adaptive_ema: float = 0.95
    adaptive_min_norm: float = 0.1


def resolve_grad_clip_config(raw: Any, default_max_norm: float = 1.0) -> GradClipConfig:
    """从 Hydra / dict 解析裁剪配置。"""
    if raw is None:
        return GradClipConfig(strategy="norm", max_norm=float(default_max_norm))
    if hasattr(raw, "get"):
        get = raw.get
    elif isinstance(raw, dict):
        get = raw.get
    else:
        return GradClipConfig(strategy="norm", max_norm=float(default_max_norm))

    adaptive = get("adaptive", {}) or {}
    if hasattr(adaptive, "get"):
        a_get = adaptive.get
    elif isinstance(adaptive, dict):
        a_get = adaptive.get
    else:
        a_get = lambda _k, d=None: d  # noqa: E731

    max_norm = get("max_norm", None)
    if max_norm is None or str(max_norm).lower() in ("null", "none", ""):
        max_norm = default_max_norm
    return GradClipConfig(
        strategy=str(get("strategy", "norm")).lower(),
        max_norm=float(max_norm),
        max_value=float(get("max_value", 1.0)),
        adaptive_window=int(a_get("window", 100)),
        adaptive_percentile=float(a_get("percentile", 90.0)),
        adaptive_ema=float(a_get("ema", 0.95)),
        adaptive_min_norm=float(a_get("min_norm", 0.1)),
    )


def _iter_params(parameters: Iterable[nn.Parameter] | nn.Module):
    if isinstance(parameters, nn.Module):
        params = [p for p in parameters.parameters() if p.grad is not None]
    else:
        params = [p for p in parameters if p.grad is not None]
    return params


def global_grad_norm(parameters: Iterable[nn.Parameter] | nn.Module) -> float:
    params = _iter_params(parameters)
    if not params:
        return 0.0
    total = torch.zeros((), device=params[0].grad.device, dtype=torch.float32)
    for p in params:
        grad = p.grad.detach()
        if grad.is_sparse:
            grad = grad.coalesce().values()
        total = total + torch.norm(grad.float(), 2.0) ** 2
    return float(torch.sqrt(total).item())


@dataclass
class GradClipper:
    """可注入 Trainer 的裁剪器。"""

    config: GradClipConfig
    _history: deque = field(default_factory=deque, init=False, repr=False)
    _ema: float | None = field(default=None, init=False, repr=False)
    last_raw_norm: float = field(default=0.0, init=False)
    last_clip_coef: float = field(default=1.0, init=False)
    last_threshold: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self._history = deque(maxlen=max(1, int(self.config.adaptive_window)))

    def clip(self, parameters: Iterable[nn.Parameter] | nn.Module) -> float:
        strategy = self.config.strategy
        if strategy in ("none", "off", "false"):
            self.last_raw_norm = global_grad_norm(parameters)
            self.last_clip_coef = 1.0
            self.last_threshold = float("inf")
            return self.last_raw_norm

        if strategy == "value":
            params = _iter_params(parameters)
            torch.nn.utils.clip_grad_value_(params, self.config.max_value)
            self.last_raw_norm = global_grad_norm(params)
            self.last_clip_coef = 1.0
            self.last_threshold = self.config.max_value
            return self.last_raw_norm

        raw = global_grad_norm(parameters)
        self.last_raw_norm = raw

        if strategy == "adaptive":
            if self._ema is None:
                self._ema = raw
            else:
                a = self.config.adaptive_ema
                self._ema = a * self._ema + (1.0 - a) * raw
            self._history.append(raw)

        threshold = self._threshold()
        self.last_threshold = threshold

        if raw <= 0:
            self.last_clip_coef = 1.0
            return 0.0

        coef = min(1.0, threshold / (raw + 1e-6))
        self.last_clip_coef = coef
        if coef < 1.0:
            for p in _iter_params(parameters):
                p.grad.detach().mul_(coef)
        return raw * coef

    def _threshold(self) -> float:
        cfg = self.config
        if cfg.strategy == "norm":
            return max(cfg.max_norm, 0.0)

        # adaptive：历史分位数与 EMA 的均值，夹在 [min_norm, max_norm]
        if len(self._history) <= 1:
            return max(cfg.max_norm, cfg.adaptive_min_norm)
        hist = sorted(self._history)
        idx = math_percentile_index(len(hist), cfg.adaptive_percentile)
        p = hist[idx]
        ema = self._ema if self._ema is not None else p
        dynamic = max(cfg.adaptive_min_norm, 0.5 * (p + ema))
        if cfg.max_norm > 0:
            return min(cfg.max_norm, dynamic)
        return dynamic

    def metrics(self) -> dict[str, float]:
        return {
            "grad/raw_norm": float(self.last_raw_norm),
            "grad/clip_coef": float(self.last_clip_coef),
            "grad/clip_threshold": float(self.last_threshold),
        }


def math_percentile_index(n: int, percentile: float) -> int:
    if n <= 1:
        return 0
    return int(round((max(0.0, min(100.0, percentile)) / 100.0) * (n - 1)))


def build_grad_clipper(raw: Any, default_max_norm: float = 1.0) -> GradClipper:
    return GradClipper(resolve_grad_clip_config(raw, default_max_norm=default_max_norm))
