"""数据流水线：load → adapt → clean → split → version → project。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

from src.data.adapters import get_adapter
from src.data.cleaning import clean_samples
from src.data.projectors import project_splits
from src.data.schema import Sample
from src.data.split import split_samples
from src.data.versioning import DatasetVersion, make_version, save_version, verify_version

logger = logging.getLogger("posttrainlab.data.pipeline")


def _load_raw_rows(cfg: Any) -> list[dict[str, Any]]:
    """从 HF hub / 本地 jsonl / 已有 split 加载原始行。"""
    source = cfg.get("source") or cfg.get("dataset_path")
    if source is None:
        raise ValueError("data.source 或 data.dataset_path 必须指定")

    source_str = str(source)
    path = Path(source_str)
    if path.suffix in {".jsonl", ".json"} and path.is_file():
        rows = []
        if path.suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
            rows = data if isinstance(data, list) else data.get("data", [])
        return rows

    # HuggingFace datasets
    ds_config = cfg.get("dataset_config")
    split_name = cfg.get("split_train") or cfg.get("hf_split") or "train"
    if ds_config is not None and str(ds_config) not in ("null", "None"):
        raw = load_dataset(source_str, str(ds_config), split=split_name)
    else:
        # 可能带 train+test；先取 train，若 pipeline.split.use_official_eval 再补
        try:
            raw = load_dataset(source_str, split=split_name)
        except Exception:
            raw = load_dataset(source_str)
            if isinstance(raw, DatasetDict):
                raw = raw[split_name]
    return [dict(raw[i]) for i in range(len(raw))]


def _maybe_cap(samples: list[Sample], n: int | None) -> list[Sample]:
    if n is None or n >= len(samples):
        return samples
    return samples[: int(n)]


def build_samples(cfg: Any) -> list[Sample]:
    fmt = str(cfg.get("format") or cfg.get("template") or "instruction")
    # 兼容旧配置 template: gsm8k_boxed → format gsm8k
    alias = {
        "gsm8k_boxed": "gsm8k",
        "sft_messages": "sharegpt",
    }
    fmt = alias.get(fmt, fmt)

    rows = _load_raw_rows(cfg)
    adapter = get_adapter(fmt)
    system_prompt = cfg.get("system_prompt")
    samples = adapter.convert_many(rows, system_prompt=system_prompt)
    samples = clean_samples(samples, cfg.get("cleaning"))
    samples = _maybe_cap(samples, cfg.get("max_train_samples"))
    return samples


def build_pipeline(
    data_cfg: Any,
    *,
    algorithm: str,
    seed: int = 12,
    version_dir: str | Path | None = None,
) -> tuple[DatasetDict, DatasetVersion]:
    """
    一份原始数据 → 统一 Sample → 清洗/划分/版本 → 投影为算法 DatasetDict。

    返回 (dataset_dict, version_meta)。
    dataset_dict 含 train / eval（及可选 test）。
    """
    samples = build_samples(data_cfg)

    split_cfg = data_cfg.get("split") or {}
    use_official = bool(split_cfg.get("use_official_eval", False))

    if use_official and data_cfg.get("split_eval"):
        # 官方 eval split：train 用已构建 samples；eval 再加载官方集
        train_samples = samples
        from omegaconf import OmegaConf

        eval_overlay = OmegaConf.create(OmegaConf.to_container(data_cfg, resolve=True))
        eval_overlay.split_train = data_cfg.split_eval
        eval_overlay.max_train_samples = data_cfg.get("max_eval_samples")
        eval_rows = _load_raw_rows(eval_overlay)
        fmt = str(data_cfg.get("format") or data_cfg.get("template") or "instruction")
        alias = {"gsm8k_boxed": "gsm8k", "sft_messages": "sharegpt"}
        fmt = alias.get(fmt, fmt)
        eval_samples = get_adapter(fmt).convert_many(
            eval_rows, system_prompt=data_cfg.get("system_prompt")
        )
        eval_samples = clean_samples(eval_samples, data_cfg.get("cleaning"))
        eval_samples = _maybe_cap(eval_samples, data_cfg.get("max_eval_samples"))
        splits = {"train": train_samples, "val": eval_samples, "test": eval_samples}
    else:
        ratios = split_cfg.get("ratios", [0.8, 0.1, 0.1])
        splits = split_samples(
            samples,
            ratios=list(ratios),
            seed=int(split_cfg.get("seed", seed)),
            stratify=bool(split_cfg.get("stratify", False)),
            stratify_key=str(split_cfg.get("stratify_key", "stratify_key")),
        )
        # max_eval_samples 限制 val
        if data_cfg.get("max_eval_samples") is not None:
            splits["val"] = _maybe_cap(splits["val"], data_cfg.get("max_eval_samples"))
            splits["test"] = _maybe_cap(splits["test"], data_cfg.get("max_eval_samples"))

    version = make_version(
        splits,
        version=data_cfg.get("version"),
        extras={
            "format": str(data_cfg.get("format") or data_cfg.get("template")),
            "algorithm": algorithm,
            "source": str(data_cfg.get("source") or data_cfg.get("dataset_path")),
        },
    )
    # 可选：实验要求绑定特定版本 / 哈希（追溯复现）
    expected_version = data_cfg.get("expected_version")
    expected_hash = data_cfg.get("expected_hash")
    if expected_version or expected_hash:
        verify_version(
            version,
            expected_version=expected_version,
            expected_hash=expected_hash,
        )

    out_dir = version_dir or data_cfg.get("version_dir") or "outputs/data_versions"
    save_version(version, out_dir)

    ds = project_splits(splits, algorithm)
    logger.info(
        "流水线完成 algo=%s version=%s hash=%s train=%d eval=%d",
        algorithm,
        version.version,
        version.hash[:12],
        len(ds["train"]) if "train" in ds else 0,
        len(ds["eval"]) if "eval" in ds else 0,
    )
    return ds, version
