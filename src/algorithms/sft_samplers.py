"""SFT 自研采样器：随机、长度分组、分桶。

长度分组思路：在 mega-batch 内按序列长度排序，使同一步内样本长度接近，
减少 padding，提高有效吞吐。分桶则把样本按长度边界划入固定桶再均匀抽样。
"""

from __future__ import annotations

import random
from typing import Any, Iterator, Sequence

try:
    from torch.utils.data import Sampler
except Exception:  # pragma: no cover - 无 torch 时仅供类型检查/文档

    class Sampler:  # type: ignore[no-redef]
        pass


def estimate_example_length(example: Any, max_length: int = 2048) -> int:
    """无 tokenizer 时的长度估计：按字符粗算 token 数。"""
    if isinstance(example, dict):
        if "length" in example and example["length"] is not None:
            return max(1, min(int(example["length"]), max_length))
        if "messages" in example and example["messages"] is not None:
            parts = []
            for msg in example["messages"]:
                if isinstance(msg, dict):
                    parts.append(str(msg.get("content", "")))
                else:
                    parts.append(str(msg))
            text = "\n".join(parts)
        elif "text" in example:
            text = str(example["text"])
        elif "prompt" in example:
            text = str(example["prompt"])
        else:
            text = str(example)
    else:
        text = str(example)
    # 中英混合粗略：约 3.5 字符 / token
    return max(1, min(int(len(text) / 3.5) + 1, max_length))


def estimate_dataset_lengths(
    dataset: Any,
    max_length: int = 2048,
    tokenizer: Any | None = None,
) -> list[int]:
    """为整份数据集预计算长度（优先用 tokenizer，否则用字符启发式）。"""
    n = len(dataset)
    lengths: list[int] = []
    for i in range(n):
        ex = dataset[i]
        if tokenizer is not None and isinstance(ex, dict):
            if "messages" in ex and hasattr(tokenizer, "apply_chat_template"):
                try:
                    text = tokenizer.apply_chat_template(
                        ex["messages"], tokenize=False, add_generation_prompt=False
                    )
                    ids = tokenizer.encode(text, add_special_tokens=False)
                    lengths.append(max(1, min(len(ids), max_length)))
                    continue
                except Exception:
                    pass
            if "text" in ex:
                ids = tokenizer.encode(str(ex["text"]), add_special_tokens=True)
                lengths.append(max(1, min(len(ids), max_length)))
                continue
        lengths.append(estimate_example_length(ex, max_length=max_length))
    return lengths


class RandomOrderSampler(Sampler):
    """可复现的随机下标采样器（单进程）。"""

    def __init__(self, num_samples: int, *, shuffle: bool = True, seed: int = 0):
        self.num_samples = int(num_samples)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        indices = list(range(self.num_samples))
        if self.shuffle:
            rng = random.Random(self.seed + self._epoch)
            rng.shuffle(indices)
        return iter(indices)


class LengthGroupedSampler(Sampler):
    """
    长度分组采样器。

    流程：打乱 → 切成 mega-batch → 每个 mega-batch 内按长度降序 → 展平。
    ``mega_batch_size`` 通常取 ``per_device_batch * grad_accum * mega_batch_mult``。
    """

    def __init__(
        self,
        lengths: Sequence[int],
        *,
        batch_size: int,
        mega_batch_mult: int = 8,
        seed: int = 0,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        if batch_size < 1:
            raise ValueError("batch_size 必须 >= 1")
        self.lengths = [int(x) for x in lengths]
        self.batch_size = int(batch_size)
        self.mega_batch_size = max(self.batch_size, self.batch_size * max(1, int(mega_batch_mult)))
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        n = len(self.lengths)
        if self.drop_last:
            return n - (n % self.batch_size)
        return n

    def __iter__(self) -> Iterator[int]:
        n = len(self.lengths)
        indices = list(range(n))
        rng = random.Random(self.seed + self._epoch)
        if self.shuffle:
            rng.shuffle(indices)

        mega = self.mega_batch_size
        grouped: list[int] = []
        for start in range(0, n, mega):
            chunk = indices[start : start + mega]
            chunk.sort(key=lambda i: self.lengths[i], reverse=True)
            grouped.extend(chunk)

        if self.drop_last:
            usable = len(grouped) - (len(grouped) % self.batch_size)
            grouped = grouped[:usable]
        return iter(grouped)


class BucketSampler(Sampler):
    """
    分桶采样器：按长度边界划桶，每步尽量从同一桶取满一个 batch。

    桶空时跳过；各桶按轮询顺序出 batch，再在桶内打乱。
    """

    def __init__(
        self,
        lengths: Sequence[int],
        *,
        batch_size: int,
        boundaries: Sequence[int] | None = None,
        seed: int = 0,
        drop_last: bool = False,
    ):
        self.lengths = [int(x) for x in lengths]
        self.batch_size = int(batch_size)
        bounds = sorted(int(b) for b in (boundaries or [128, 256, 512, 1024, 2048]))
        self.boundaries = bounds
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self._epoch = 0
        self._buckets = self._assign_buckets()

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _bucket_id(self, length: int) -> int:
        for i, b in enumerate(self.boundaries):
            if length <= b:
                return i
        return len(self.boundaries)

    def _assign_buckets(self) -> list[list[int]]:
        buckets: list[list[int]] = [[] for _ in range(len(self.boundaries) + 1)]
        for i, length in enumerate(self.lengths):
            buckets[self._bucket_id(length)].append(i)
        return buckets

    def __len__(self) -> int:
        total = 0
        for bucket in self._buckets:
            if self.drop_last:
                total += len(bucket) - (len(bucket) % self.batch_size)
            else:
                total += len(bucket)
        return total

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self._epoch)
        # 每桶独立打乱后切 batch
        batches: list[list[int]] = []
        for bucket in self._buckets:
            if not bucket:
                continue
            order = list(bucket)
            rng.shuffle(order)
            for start in range(0, len(order), self.batch_size):
                chunk = order[start : start + self.batch_size]
                if len(chunk) < self.batch_size and self.drop_last:
                    continue
                batches.append(chunk)
        rng.shuffle(batches)
        for batch in batches:
            yield from batch


def build_sft_sampler(
    dataset: Any,
    *,
    mode: str = "length_group",
    batch_size: int = 1,
    mega_batch_mult: int = 8,
    boundaries: Sequence[int] | None = None,
    seed: int = 0,
    max_length: int = 2048,
    tokenizer: Any | None = None,
    lengths: Sequence[int] | None = None,
    drop_last: bool = False,
) -> Sampler:
    """按配置构造采样器。"""
    mode = str(mode).lower()
    n = len(dataset)
    if mode in ("random", "default", "none"):
        return RandomOrderSampler(n, shuffle=True, seed=seed)

    lens = list(lengths) if lengths is not None else estimate_dataset_lengths(
        dataset, max_length=max_length, tokenizer=tokenizer
    )
    if len(lens) != n:
        raise ValueError(f"lengths 长度 {len(lens)} 与数据集 {n} 不一致")

    if mode in ("length_group", "length", "group_by_length"):
        return LengthGroupedSampler(
            lens,
            batch_size=batch_size,
            mega_batch_mult=mega_batch_mult,
            seed=seed,
            drop_last=drop_last,
        )
    if mode in ("bucket", "buckets"):
        return BucketSampler(
            lens,
            batch_size=batch_size,
            boundaries=boundaries,
            seed=seed,
            drop_last=drop_last,
        )
    raise ValueError(f"未知 sampler.mode={mode!r}，可选: random | length_group | bucket")


def length_group_waste_ratio(lengths: Sequence[int], order: Sequence[int], batch_size: int) -> float:
    """评估采样顺序的 padding 浪费率（越大越差），便于单测。"""
    if batch_size < 1 or not order:
        return 0.0
    waste = 0.0
    tokens = 0.0
    for start in range(0, len(order), batch_size):
        batch = order[start : start + batch_size]
        if not batch:
            continue
        batch_lens = [lengths[i] for i in batch]
        maxlen = max(batch_lens)
        waste += sum(maxlen - L for L in batch_lens)
        tokens += maxlen * len(batch_lens)
    return waste / tokens if tokens else 0.0


def suggest_mega_batch_size(batch_size: int, grad_accum: int, mult: int, world_size: int = 1) -> int:
    """推荐 mega-batch = 微批 × 累积 × 倍数 × 卡数。"""
    return max(1, int(batch_size) * max(1, int(grad_accum)) * max(1, int(mult)) * max(1, int(world_size)))
