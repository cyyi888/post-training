from __future__ import annotations

from collections.abc import Callable
from typing import Any

ALGORITHM_REGISTRY: dict[str, type] = {}
DATA_REGISTRY: dict[str, Callable[..., Any]] = {}


def register(registry: dict[str, Any], name: str):
    """Decorator to register trainers / datasets / reward funcs."""

    def decorator(obj: Any):
        if name in registry:
            raise ValueError(f"Duplicate registry key: {name}")
        registry[name] = obj
        return obj

    return decorator
