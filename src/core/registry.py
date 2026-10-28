from __future__ import annotations

ALGORITHM_REGISTRY: dict[str, type] = {}
EVALUATOR_REGISTRY: dict[str, type] = {}


def register(registry: dict[str, object], name: str):
    """Decorator to register trainers / evaluators / reward funcs."""

    def decorator(obj: object):
        if name in registry:
            raise ValueError(f"Duplicate registry key: {name}")
        registry[name] = obj
        return obj

    return decorator
