"""数据集版本：内容哈希 + 版本号，供实验绑定追溯。"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.data.schema import Sample

logger = logging.getLogger("posttrainlab.data.versioning")


@dataclass
class DatasetVersion:
    """一次数据处理结果的可追溯元信息。"""

    version: str
    hash: str
    num_samples: int
    split_sizes: dict[str, int]
    created_at: str
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetVersion:
        return cls(
            version=str(data["version"]),
            hash=str(data["hash"]),
            num_samples=int(data["num_samples"]),
            split_sizes=dict(data.get("split_sizes") or {}),
            created_at=str(data.get("created_at") or ""),
            extras=dict(data.get("extras") or {}),
        )

    def short_hash(self, n: int = 12) -> str:
        return self.hash[:n]


def compute_samples_hash(samples: list[Sample]) -> str:
    """对样本内容哈希排序后聚合，顺序无关、可复现。"""
    h = hashlib.sha256()
    for s in sorted(samples, key=lambda x: x.content_hash()):
        h.update(s.content_hash().encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def make_version(
    splits: dict[str, list[Sample]],
    version: str | None = None,
    extras: dict[str, Any] | None = None,
) -> DatasetVersion:
    all_samples: list[Sample] = []
    sizes: dict[str, int] = {}
    for name, rows in splits.items():
        sizes[name] = len(rows)
        all_samples.extend(rows)
    digest = compute_samples_hash(all_samples)
    if not version:
        version = f"v-{digest[:8]}"
    return DatasetVersion(
        version=str(version),
        hash=digest,
        num_samples=len(all_samples),
        split_sizes=sizes,
        created_at=datetime.now(timezone.utc).isoformat(),
        extras=dict(extras or {}),
    )


def version_dir_for(out_dir: str | Path, meta: DatasetVersion) -> Path:
    """outputs/data_versions/<name>/v-xxxxxxxx/"""
    return Path(out_dir) / meta.version


def save_version(
    meta: DatasetVersion,
    out_dir: str | Path,
    *,
    use_version_subdir: bool = True,
) -> Path:
    """
    写入 dataset_version.json。

    - use_version_subdir=True：落到 ``out_dir/<version>/dataset_version.json``
      并额外在 out_dir 写一份 ``latest.json`` 指针，便于实验快速引用。
    - False：直接写 ``out_dir/dataset_version.json``（兼容旧行为）。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if use_version_subdir:
        target = version_dir_for(out_dir, meta)
        target.mkdir(parents=True, exist_ok=True)
        path = target / "dataset_version.json"
        latest = out_dir / "latest.json"
        latest.write_text(
            json.dumps(
                {"version": meta.version, "hash": meta.hash, "path": str(path)},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    else:
        path = out_dir / "dataset_version.json"

    path.write_text(
        json.dumps(meta.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(
        "数据版本已写入 %s (version=%s hash=%s)",
        path,
        meta.version,
        meta.short_hash(),
    )
    return path


def load_version(path: str | Path) -> DatasetVersion:
    """从 dataset_version.json 或目录（含该文件 / latest.json）加载。"""
    path = Path(path)
    if path.is_dir():
        candidate = path / "dataset_version.json"
        if not candidate.is_file():
            latest = path / "latest.json"
            if latest.is_file():
                ptr = json.loads(latest.read_text(encoding="utf-8"))
                candidate = Path(ptr["path"])
            else:
                raise FileNotFoundError(f"目录中无 dataset_version.json / latest.json: {path}")
        path = candidate
    data = json.loads(path.read_text(encoding="utf-8"))
    return DatasetVersion.from_dict(data)


def verify_version(
    meta: DatasetVersion,
    *,
    expected_version: str | None = None,
    expected_hash: str | None = None,
    hash_prefix: bool = True,
) -> None:
    """
    校验实验绑定的数据版本；不匹配则抛 ValueError。

    expected_hash 可写完整 sha256，或仅前缀（hash_prefix=True）。
    """
    if expected_version and str(expected_version) != meta.version:
        raise ValueError(
            f"数据 version 不匹配: 期望 {expected_version!r}，实际 {meta.version!r}"
        )
    if expected_hash:
        exp = str(expected_hash).lower()
        got = meta.hash.lower()
        ok = got.startswith(exp) if hash_prefix else got == exp
        if not ok:
            raise ValueError(
                f"数据 hash 不匹配: 期望 {expected_hash!r}，实际 {meta.short_hash()}…"
            )


def list_versions(out_dir: str | Path) -> list[DatasetVersion]:
    """列出 out_dir 下各版本子目录中的 DatasetVersion（按 created_at 排序）。"""
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return []
    versions: list[DatasetVersion] = []
    for child in sorted(out_dir.iterdir()):
        meta_path = child / "dataset_version.json" if child.is_dir() else None
        if meta_path and meta_path.is_file():
            versions.append(load_version(meta_path))
        elif child.name == "dataset_version.json" and child.is_file():
            versions.append(load_version(child))
    versions.sort(key=lambda v: v.created_at)
    return versions
