"""【设备：CPU】SFT→DPO→RL 链路编排单测。

原因：只校验 override 展开与 dry_run，不加载模型、不跑训练。
"""

from __future__ import annotations

from pathlib import Path

import pytest

OmegaConf = pytest.importorskip("omegaconf").OmegaConf


def test_build_stage_cli_overrides_injects_init_and_rl():
    from scripts.pipeline import build_stage_cli_overrides

    cfg = OmegaConf.create(
        {
            "output_dir": "outputs",
            "seed": 12,
            "use_gpu": True,
            "wandb": {"enabled": False, "mode": "disabled"},
            "pipeline": {
                "max_steps": 20,
                "rl_algorithm": "grpo",
                "stages": [],
            },
        }
    )
    dpo = OmegaConf.create(
        {
            "name": "dpo",
            "init_from": "sft",
            "overrides": ["training=dpo", "data=dpo_demo"],
        }
    )
    ov = build_stage_cli_overrides(cfg, dpo)
    assert "training=dpo" in ov
    assert "engineering.init_from_checkpoint=sft" in ov
    assert "training.max_steps=20" in ov
    assert "wandb=disabled" in ov

    rl = OmegaConf.create({"name": "rl", "init_from": "dpo", "overrides": ["data=gsm8k"]})
    ov_rl = build_stage_cli_overrides(cfg, rl)
    assert "training=grpo" in ov_rl
    assert "engineering.init_from_checkpoint=dpo" in ov_rl

    cfg.pipeline.rl_algorithm = "ppo"
    ov_ppo = build_stage_cli_overrides(cfg, rl)
    assert "training=ppo" in ov_ppo


def test_pipeline_dry_run(tmpdir):
    from scripts.pipeline import run_pipeline

    cfg = OmegaConf.create(
        {
            "output_dir": str(tmpdir),
            "seed": 1,
            "use_gpu": False,
            "wandb": {"enabled": False, "mode": "disabled"},
            "pipeline": {
                "name": "toy_chain",
                "max_steps": 2,
                "rl_algorithm": "grpo",
                "dry_run": True,
                "stages": [
                    {"name": "sft", "overrides": ["training=sft", "data=gsm8k"]},
                    {
                        "name": "dpo",
                        "init_from": "sft",
                        "overrides": ["training=dpo", "data=dpo_demo"],
                    },
                    {"name": "rl", "init_from": "dpo", "overrides": ["data=gsm8k"]},
                ],
            },
        }
    )
    results = run_pipeline(cfg)
    assert len(results) == 3
    assert all(r["status"] == "dry_run" for r in results)
    assert results[2]["overrides"][0] == "training=grpo"
    summary = Path(str(tmpdir)) / "pipeline" / "toy_chain" / "summary.yaml"
    assert summary.is_file()
