#!/usr/bin/env python
"""SFT → DPO → RL 一键链路：顺序调用 ``scripts/train.py``，用 init_from 衔接权重。"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger("posttrainlab.pipeline")


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(x) for x in value]


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def build_stage_cli_overrides(cfg: Any, stage: Any) -> list[str]:
    """把 pipeline.stages[*] 展开为传给 ``train.py`` 的 Hydra override 列表。"""
    pipe = _cfg_get(cfg, "pipeline")
    name = str(_cfg_get(stage, "name", "stage"))
    overrides = _as_list(_cfg_get(stage, "overrides"))
    rl_algo = str(_cfg_get(pipe, "rl_algorithm", "grpo")).lower()

    if name == "rl":
        if rl_algo == "ppo":
            overrides = ["training=ppo", *overrides]
        else:
            overrides = ["training=grpo", *overrides]

    init_from = _cfg_get(stage, "init_from")
    if init_from not in (None, "", "null", "none", False):
        overrides.append(f"engineering.init_from_checkpoint={init_from}")

    max_steps = _cfg_get(pipe, "max_steps")
    if max_steps not in (None, "null"):
        overrides.append(f"training.max_steps={int(max_steps)}")

    # 透传常用全局项，保证链路目录与单次 train 一致
    output_dir = _cfg_get(cfg, "output_dir")
    if output_dir:
        overrides.append(f"output_dir={output_dir}")
    seed = _cfg_get(cfg, "seed")
    if seed is not None:
        overrides.append(f"seed={int(seed)}")
    use_gpu = _cfg_get(cfg, "use_gpu")
    if use_gpu is not None:
        overrides.append(f"use_gpu={str(use_gpu).lower()}")
    wandb = _cfg_get(cfg, "wandb")
    if wandb is not None:
        mode = _cfg_get(wandb, "mode")
        enabled = _cfg_get(wandb, "enabled")
        if enabled is False or str(mode) in ("disabled", "null", "None"):
            overrides.append("wandb=disabled")
        elif str(mode) == "offline":
            overrides.append("wandb=offline")
        elif enabled or str(mode) == "online":
            overrides.append("wandb=online")
    return overrides


def run_pipeline(cfg: Any) -> list[dict[str, Any]]:
    """按 stages 顺序跑训练；``dry_run`` 时只返回将执行的命令。"""
    from omegaconf import OmegaConf

    pipe = _cfg_get(cfg, "pipeline")
    stages = list(_cfg_get(pipe, "stages"))
    dry = bool(_cfg_get(pipe, "dry_run", False))
    train_script = ROOT / "scripts" / "train.py"
    results: list[dict[str, Any]] = []

    for stage in stages:
        name = str(_cfg_get(stage, "name", "stage"))
        overrides = build_stage_cli_overrides(cfg, stage)
        cmd = [sys.executable, str(train_script), *overrides]
        logger.info("=== pipeline stage=%s ===", name)
        logger.info("cmd: %s", " ".join(cmd))
        row: dict[str, Any] = {"stage": name, "cmd": cmd, "overrides": overrides}
        if dry:
            row["status"] = "dry_run"
            results.append(row)
            continue
        proc = subprocess.run(cmd, cwd=str(ROOT), check=False)
        row["returncode"] = int(proc.returncode)
        row["status"] = "ok" if proc.returncode == 0 else "failed"
        results.append(row)
        if proc.returncode != 0:
            raise RuntimeError(f"pipeline 阶段 {name} 失败，returncode={proc.returncode}")

    out_dir = Path(str(_cfg_get(cfg, "output_dir", "outputs"))) / "pipeline" / str(
        _cfg_get(pipe, "name", "sft_dpo_rl")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = out_dir / "summary.yaml"
    OmegaConf.save(OmegaConf.create(results), summary)
    logger.info("pipeline 完成 → %s", summary)
    return results


def main() -> None:
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../configs", config_name="experiment/sft_dpo_rl")
    def _hydra_main(cfg: DictConfig) -> None:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        run_pipeline(cfg)

    _hydra_main()


if __name__ == "__main__":
    main()
