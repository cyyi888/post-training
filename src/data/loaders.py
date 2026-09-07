from __future__ import annotations

from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

from src.core.registry import DATA_REGISTRY, register
from src.data.templates import apply_template


def _maybe_select(ds: Dataset, n: int | None) -> Dataset:
    if n is None or n >= len(ds):
        return ds
    return ds.select(range(n))


@register(DATA_REGISTRY, "gsm8k")
def load_gsm8k(cfg: Any) -> DatasetDict:
    raw = load_dataset(cfg.dataset_path, cfg.dataset_config)

    def _map(example: dict) -> dict:
        return apply_template(cfg.template, example, system_prompt=cfg.system_prompt)

    train = raw[cfg.split_train].map(_map)
    eval_ds = raw[cfg.split_eval].map(_map)
    train = _maybe_select(train, cfg.get("max_train_samples"))
    eval_ds = _maybe_select(eval_ds, cfg.get("max_eval_samples"))

    drop = [c for c in ("question", "answer") if c in train.column_names]
    if drop:
        train = train.remove_columns(drop)
        eval_ds = eval_ds.remove_columns(drop)
    return DatasetDict(train=train, eval=eval_ds)


@register(DATA_REGISTRY, "humaneval")
def load_humaneval(cfg: Any) -> DatasetDict:
    raw = load_dataset(cfg.dataset_path)
    split = cfg.split_eval or "test"
    eval_ds = raw[split]

    def _map(example: dict) -> dict:
        return apply_template(cfg.template, example, system_prompt=cfg.system_prompt)

    eval_ds = eval_ds.map(_map)
    eval_ds = _maybe_select(eval_ds, cfg.get("max_eval_samples"))
    return DatasetDict(train=eval_ds, eval=eval_ds)


def load_task_dataset(cfg: Any) -> DatasetDict:
    name = cfg.name
    if name not in DATA_REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Registered: {list(DATA_REGISTRY)}")
    return DATA_REGISTRY[name](cfg)


def load_preference_dataset(
    path: str = "banghua/DL-DPO-Dataset",
    split: str = "train",
    max_samples: int | None = 256,
) -> Dataset:
    ds = load_dataset(path, split=split)
    return _maybe_select(ds, max_samples)
