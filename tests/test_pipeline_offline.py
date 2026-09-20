"""离线流水线：本地 JSONL → Sample → 版本 → 投影（无外网）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

OmegaConf = pytest.importorskip("omegaconf").OmegaConf
pytest.importorskip("datasets")

from src.data.pipeline import build_pipeline
from src.data.versioning import load_version, verify_version


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )


def test_pipeline_instruction_to_sft_with_version(tmpdir):
    tmp_path = Path(str(tmpdir))
    data_path = tmp_path / "sft.jsonl"
    _write_jsonl(
        data_path,
        [
            {"instruction": "问1", "input": "", "output": "答1"},
            {"instruction": "问2", "input": "补充", "output": "答2"},
            {"instruction": "问3", "output": "答3"},
            {"instruction": "问4", "output": "答4"},
            {"instruction": "问5", "output": "答5"},
            {"instruction": "问6", "output": "答6"},
            {"instruction": "问7", "output": "答7"},
            {"instruction": "问8", "output": "答8"},
            {"instruction": "问9", "output": "答9"},
            {"instruction": "问10", "output": "答10"},
        ],
    )
    version_root = tmp_path / "versions"
    cfg = OmegaConf.create(
        {
            "name": "toy_sft",
            "format": "instruction",
            "source": str(data_path),
            "system_prompt": "sys",
            "max_train_samples": None,
            "max_eval_samples": None,
            "version": "v-toy",
            "split": {"use_official_eval": False, "ratios": [0.7, 0.2, 0.1], "seed": 12},
            "cleaning": {"dedup": True, "min_chars": 1, "filter_low_quality": True},
        }
    )
    ds, meta = build_pipeline(
        cfg, algorithm="sft", seed=12, version_dir=version_root
    )
    assert meta.version == "v-toy"
    assert "train" in ds and len(ds["train"]) > 0
    assert "messages" in ds["train"][0]

    saved = load_version(version_root)
    assert saved.hash == meta.hash
    verify_version(saved, expected_version="v-toy", expected_hash=meta.hash[:12])


def test_pipeline_expected_hash_mismatch(tmpdir):
    tmp_path = Path(str(tmpdir))
    data_path = tmp_path / "dpo.jsonl"
    _write_jsonl(
        data_path,
        [
            {
                "prompt": "Name?",
                "chosen": "A",
                "rejected": "B",
            }
        ]
        * 5,
    )
    cfg = OmegaConf.create(
        {
            "format": "dpo_pair",
            "source": str(data_path),
            "expected_hash": "0" * 64,
            "split": {"ratios": [1.0, 0.0, 0.0], "seed": 1},
            "cleaning": {"dedup": True, "min_chars": 1},
        }
    )
    with pytest.raises(ValueError, match="hash"):
        build_pipeline(cfg, algorithm="dpo", seed=1, version_dir=tmp_path / "v")
