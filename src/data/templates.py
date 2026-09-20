"""兼容旧任务模板名；新代码请用 adapters + Sample。"""

from __future__ import annotations

from typing import Any

from src.data.adapters import get_adapter
from src.data.schema import Sample

# 旧名 → 新 format
_ALIAS = {
    "gsm8k_boxed": "gsm8k",
    "humaneval": "humaneval",
    "sft_messages": "sharegpt",
}


def get_template(name: str):
    fmt = _ALIAS.get(name, name)

    def _fn(example: dict, **kwargs: Any) -> dict:
        sample: Sample = get_adapter(fmt).convert(example, **kwargs)
        if fmt == "gsm8k":
            prompt = []
            if sample.system:
                prompt.append({"role": "system", "content": sample.system})
            prompt.append({"role": "user", "content": sample.prompt})
            return {"prompt": prompt, "ground_truth": sample.metadata.get("ground_truth")}
        if fmt == "humaneval":
            prompt = []
            if sample.system:
                prompt.append({"role": "system", "content": sample.system})
            prompt.append({"role": "user", "content": sample.prompt})
            return {
                "prompt": prompt,
                "ground_truth": sample.metadata.get("ground_truth"),
                "entry_point": sample.metadata.get("entry_point"),
                "test": sample.metadata.get("test"),
            }
        return {"messages": sample.to_messages()}

    return _fn


def apply_template(name: str, example: dict, **kwargs: Any) -> dict:
    return get_template(name)(example, **kwargs)
