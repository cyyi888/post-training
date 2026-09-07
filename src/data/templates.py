from __future__ import annotations

import re
from typing import Any, Callable

TEMPLATES: dict[str, Callable[..., dict]] = {}


def get_template(name: str) -> Callable[..., dict]:
    if name not in TEMPLATES:
        raise KeyError(f"Unknown template '{name}'. Available: {list(TEMPLATES)}")
    return TEMPLATES[name]


def apply_template(name: str, example: dict, **kwargs: Any) -> dict:
    return get_template(name)(example, **kwargs)


def _register(name: str):
    def deco(fn: Callable[..., dict]):
        TEMPLATES[name] = fn
        return fn

    return deco


@_register("gsm8k_boxed")
def gsm8k_boxed(example: dict, system_prompt: str = "", **_: Any) -> dict:
    match = re.search(r"####\s*(-?\d+)", example.get("answer", ""))
    ground_truth = match.group(1) if match else None
    prompt = []
    if system_prompt:
        prompt.append({"role": "system", "content": system_prompt})
    prompt.append({"role": "user", "content": example["question"]})
    return {"prompt": prompt, "ground_truth": ground_truth}


@_register("humaneval")
def humaneval(example: dict, system_prompt: str = "", **_: Any) -> dict:
    prompt = []
    if system_prompt:
        prompt.append({"role": "system", "content": system_prompt})
    prompt.append({"role": "user", "content": example.get("prompt", "")})
    return {
        "prompt": prompt,
        "ground_truth": example.get("canonical_solution"),
        "entry_point": example.get("entry_point"),
        "test": example.get("test"),
    }


@_register("sft_messages")
def sft_messages(example: dict, **_: Any) -> dict:
    """Passthrough for datasets already in chat `messages` format."""
    return {"messages": example["messages"]}
