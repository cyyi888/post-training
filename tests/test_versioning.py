"""数据集版本管理单元测试（无 GPU / 外网）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.schema import Sample
from src.data.versioning import (
    DatasetVersion,
    compute_samples_hash,
    list_versions,
    load_version,
    make_version,
    save_version,
    verify_version,
)


def _samples() -> list[Sample]:
    return [
        Sample(prompt="q1", chosen="a1"),
        Sample(prompt="q2", chosen="a2", rejected="b2"),
    ]


def test_hash_order_invariant():
    s = _samples()
    assert compute_samples_hash(s) == compute_samples_hash(list(reversed(s)))


def test_hash_changes_when_content_changes():
    s = _samples()
    h0 = compute_samples_hash(s)
    s2 = [Sample(prompt="q1", chosen="CHANGED"), s[1]]
    assert compute_samples_hash(s2) != h0


def test_make_version_auto_name_and_sizes():
    splits = {"train": _samples(), "val": [Sample(prompt="v", chosen="c")], "test": []}
    meta = make_version(splits, extras={"format": "instruction"})
    assert meta.version.startswith("v-")
    assert len(meta.version) == 10  # v- + 8 hex
    assert meta.num_samples == 3
    assert meta.split_sizes == {"train": 2, "val": 1, "test": 0}
    assert meta.extras["format"] == "instruction"
    assert len(meta.hash) == 64


def test_make_version_custom_name():
    meta = make_version({"train": _samples()}, version="v-demo")
    assert meta.version == "v-demo"


def test_save_load_roundtrip(tmpdir):
    tmp_path = Path(str(tmpdir))
    meta = make_version({"train": _samples(), "val": [], "test": []}, version="v-round")
    path = save_version(meta, tmp_path, use_version_subdir=True)
    assert path.is_file()
    assert (tmp_path / "latest.json").is_file()

    loaded = load_version(path)
    assert loaded.version == meta.version
    assert loaded.hash == meta.hash
    assert loaded.num_samples == meta.num_samples
    assert loaded.split_sizes == meta.split_sizes

    # 从父目录 latest 指针加载
    loaded2 = load_version(tmp_path)
    assert loaded2.hash == meta.hash


def test_save_flat_compat(tmpdir):
    tmp_path = Path(str(tmpdir))
    meta = make_version({"train": _samples()}, version="v-flat")
    path = save_version(meta, tmp_path, use_version_subdir=False)
    assert path == tmp_path / "dataset_version.json"
    loaded = load_version(tmp_path)
    assert loaded.version == "v-flat"


def test_verify_version_ok_and_fail():
    meta = make_version({"train": _samples()}, version="v-abc")
    verify_version(meta, expected_version="v-abc", expected_hash=meta.hash[:8])
    verify_version(meta, expected_hash=meta.hash)

    with pytest.raises(ValueError, match="version"):
        verify_version(meta, expected_version="v-other")
    with pytest.raises(ValueError, match="hash"):
        verify_version(meta, expected_hash="deadbeef" * 8)


def test_list_versions(tmpdir):
    tmp_path = Path(str(tmpdir))
    m1 = make_version({"train": [Sample(prompt="a", chosen="1")]}, version="v-001")
    m2 = make_version({"train": [Sample(prompt="b", chosen="2")]}, version="v-002")
    save_version(m1, tmp_path)
    save_version(m2, tmp_path)
    found = list_versions(tmp_path)
    assert {v.version for v in found} >= {"v-001", "v-002"}


def test_from_dict_json_shape(tmpdir):
    tmp_path = Path(str(tmpdir))
    meta = make_version({"train": _samples()}, version="v-json")
    save_version(meta, tmp_path)
    raw = json.loads((tmp_path / "v-json" / "dataset_version.json").read_text(encoding="utf-8"))
    assert set(raw.keys()) >= {"version", "hash", "num_samples", "split_sizes", "created_at"}
    again = DatasetVersion.from_dict(raw)
    assert again.hash == meta.hash
