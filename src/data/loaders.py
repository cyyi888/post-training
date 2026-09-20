"""数据加载入口：兼容旧 API，内部走统一流水线。"""

from __future__ import annotations

from typing import Any

from datasets import Dataset, DatasetDict

from src.core.registry import DATA_REGISTRY, register
from src.data.pipeline import build_pipeline
from src.data.projectors import project_dpo, project_grpo, project_sft
from src.data.schema import Sample


@register(DATA_REGISTRY, "gsm8k")
def load_gsm8k(cfg: Any) -> DatasetDict:
    from omegaconf import OmegaConf

    overlay = OmegaConf.merge(
        OmegaConf.create(
            {
                "format": "gsm8k",
                "split": {"use_official_eval": True},
            }
        ),
        cfg,
    )
    ds, _ = build_pipeline(overlay, algorithm="grpo", seed=12)
    return ds


@register(DATA_REGISTRY, "humaneval")
def load_humaneval(cfg: Any) -> DatasetDict:
    from omegaconf import OmegaConf

    overlay = OmegaConf.merge(
        OmegaConf.create({"format": "humaneval", "split": {"use_official_eval": True}}),
        cfg,
    )
    ds, _ = build_pipeline(overlay, algorithm="grpo", seed=12)
    return ds


def load_task_dataset(cfg: Any, algorithm: str = "grpo", seed: int = 12) -> DatasetDict:
    """通用入口：统一流水线并投影到指定算法。"""
    ds, _ = build_pipeline(cfg, algorithm=algorithm, seed=seed)
    return ds


def load_preference_dataset(
    path: str = "banghua/DL-DPO-Dataset",
    split: str = "train",
    max_samples: int | None = 256,
    system_prompt: str | None = None,
) -> Dataset:
    """兼容旧接口：HF 偏好数据 → Sample → DPO 投影。"""
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "source": path,
            "dataset_path": path,
            "format": "dpo_pair",
            "split_train": split,
            "max_train_samples": max_samples,
            "system_prompt": system_prompt,
            "split": {"ratios": [1.0, 0.0, 0.0], "seed": 12},
            "cleaning": {"dedup": True, "filter_low_quality": True},
        }
    )
    ds, _ = build_pipeline(cfg, algorithm="dpo", seed=12)
    return ds["train"]


def samples_to_algorithm(samples: list, algorithm: str) -> Dataset:
    typed = [s if isinstance(s, Sample) else Sample.from_dict(s) for s in samples]
    if algorithm == "sft":
        return project_sft(typed)
    if algorithm == "dpo":
        return project_dpo(typed)
    if algorithm == "grpo":
        return project_grpo(typed)
    raise KeyError(algorithm)
