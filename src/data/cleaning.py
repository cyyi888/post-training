"""数据清洗：去重、长短过滤、低质量、敏感词。"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Sequence

from src.data.schema import Sample

logger = logging.getLogger("posttrainlab.data.cleaning")


def _text_len(sample: Sample) -> int:
    parts = [sample.prompt, sample.chosen or "", sample.rejected or ""]
    return sum(len(p) for p in parts)


def deduplicate(samples: list[Sample]) -> list[Sample]:
    seen: set[str] = set()
    out: list[Sample] = []
    for s in samples:
        h = s.content_hash()
        if h in seen:
            continue
        seen.add(h)
        out.append(s)
    return out


def filter_length(
    samples: list[Sample],
    min_chars: int = 1,
    max_chars: int = 100_000,
) -> list[Sample]:
    out = []
    for s in samples:
        n = _text_len(s)
        if n < min_chars or n > max_chars:
            continue
        if not str(s.prompt).strip():
            continue
        out.append(s)
    return out


def filter_low_quality(samples: list[Sample]) -> list[Sample]:
    """过滤空回答、几乎全是重复字符、不可见乱码比例过高等。"""
    out: list[Sample] = []
    for s in samples:
        text = " ".join(
            x for x in [s.prompt, s.chosen or "", s.rejected or ""] if x
        )
        if not text.strip():
            continue
        # 重复字符占比过高
        if len(set(text.replace(" ", ""))) <= 2 and len(text) > 20:
            continue
        # 控制字符过多
        ctrl = sum(1 for c in text if ord(c) < 32 and c not in "\n\t\r")
        if len(text) > 0 and ctrl / len(text) > 0.05:
            continue
        out.append(s)
    return out


def load_sensitive_words(path: str | Path | None) -> list[str]:
    if not path:
        return []
    p = Path(path)
    if not p.is_file():
        logger.warning("敏感词文件不存在: %s", p)
        return []
    words = []
    for line in p.read_text(encoding="utf-8").splitlines():
        w = line.strip()
        if w and not w.startswith("#"):
            words.append(w)
    return words


def filter_sensitive(samples: list[Sample], words: Sequence[str]) -> list[Sample]:
    if not words:
        return samples
    pattern = re.compile("|".join(re.escape(w) for w in words), re.IGNORECASE)
    out = []
    for s in samples:
        blob = f"{s.prompt}\n{s.chosen or ''}\n{s.rejected or ''}"
        if pattern.search(blob):
            continue
        out.append(s)
    return out


def clean_samples(samples: list[Sample], cfg: Any | None = None) -> list[Sample]:
    """按配置链式清洗；cfg 为空则只做基础去重+空 prompt 过滤。"""
    cfg = cfg or {}
    n0 = len(samples)

    if cfg.get("dedup", True):
        samples = deduplicate(samples)
    samples = filter_length(
        samples,
        min_chars=int(cfg.get("min_chars", 1)),
        max_chars=int(cfg.get("max_chars", 100_000)),
    )
    if cfg.get("filter_low_quality", True):
        samples = filter_low_quality(samples)
    if cfg.get("filter_sensitive", False):
        words = load_sensitive_words(cfg.get("sensitive_words_path"))
        samples = filter_sensitive(samples, words)

    logger.info("清洗完成: %d → %d", n0, len(samples))
    return samples
