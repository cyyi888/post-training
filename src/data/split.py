"""按比例划分 train/val/test，支持分层与固定种子。"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import Any

from src.data.schema import Sample

logger = logging.getLogger("posttrainlab.data.split")


def _ratios(ratios: tuple[float, float, float] | list[float]) -> tuple[float, float, float]:
    t, v, e = float(ratios[0]), float(ratios[1]), float(ratios[2])
    s = t + v + e
    if abs(s - 1.0) > 1e-6:
        t, v, e = t / s, v / s, e / s
    return t, v, e


def _split_indices(n: int, ratios: tuple[float, float, float], rng: random.Random):
    idx = list(range(n))
    rng.shuffle(idx)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    # 极小数据集兜底：保证 train 非空
    if n > 0 and not train_idx:
        train_idx = [idx[0]]
        val_idx = [i for i in idx[1:] if i not in train_idx][: max(0, n_val)]
        used = set(train_idx) | set(val_idx)
        test_idx = [i for i in idx if i not in used]
    return train_idx, val_idx, test_idx


def split_samples(
    samples: list[Sample],
    ratios: tuple[float, float, float] | list[float] = (0.8, 0.1, 0.1),
    seed: int = 12,
    stratify: bool = False,
    stratify_key: str = "stratify_key",
) -> dict[str, list[Sample]]:
    """
    返回 {"train", "val", "test"}。

    stratify=True 时按 metadata[stratify_key] 分层；缺失键归为 "_none"。
    """
    ratios = _ratios(ratios)
    rng = random.Random(seed)

    if not samples:
        return {"train": [], "val": [], "test": []}

    if not stratify:
        tr, va, te = _split_indices(len(samples), ratios, rng)
        out = {
            "train": [samples[i] for i in tr],
            "val": [samples[i] for i in va],
            "test": [samples[i] for i in te],
        }
    else:
        buckets: dict[str, list[Sample]] = defaultdict(list)
        for s in samples:
            key = str(s.metadata.get(stratify_key, "_none"))
            buckets[key].append(s)
        out = {"train": [], "val": [], "test": []}
        for _, group in buckets.items():
            tr, va, te = _split_indices(len(group), ratios, rng)
            out["train"].extend(group[i] for i in tr)
            out["val"].extend(group[i] for i in va)
            out["test"].extend(group[i] for i in te)
        # 再打乱各 split，避免 bucket 顺序偏差
        for k in out:
            rng.shuffle(out[k])

    logger.info(
        "划分完成 seed=%s train/val/test = %d/%d/%d",
        seed,
        len(out["train"]),
        len(out["val"]),
        len(out["test"]),
    )
    return out
