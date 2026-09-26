#!/usr/bin/env python
"""消融实验入口（基于 Hydra）。"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment.scheduler import AblationScheduler
from src.utils.runtime import prepare_runtime, teardown_runtime
from src.utils.wandb_utils import finish_wandb, init_wandb, log_metrics

logger = logging.getLogger("posttrainlab.ablation")


def _dry_run(cfg, variant_name: str) -> dict:
    """	config fingerprint：只记录关键配置字段，不实际占用 GPU。"""
    t = cfg.get("training", {})
    metrics = {
        "loss_type": str(t.get("loss_type", "")),
        "beta": float(t.get("beta", 0.0) or 0.0),
        "dynamic_beta": bool(t.get("dynamic_beta", False)),
        "penalty_enabled": bool(t.get("penalty", {}).get("enabled", False))
        if t.get("penalty")
        else False,
        "status": "dry_run",
    }
    log_metrics({f"ablation/{variant_name}/{k}": v for k, v in metrics.items() if isinstance(v, (int, float, bool))})
    return metrics


@hydra.main(version_base=None, config_path="../configs", config_name="experiment/ablation")
def main(cfg: DictConfig) -> None:
    track = bool(cfg.experiment.get("track_wandb", False))
    if track:
        OmegaConf.set_struct(cfg, False)
        cfg.wandb.enabled = True
        if str(cfg.wandb.get("mode", "disabled")) == "disabled":
            cfg.wandb.mode = "online"

    prepare_runtime(cfg, init_wb=track)
    try:
        logger.info("Ablation: %s", cfg.experiment.name)
        scheduler = AblationScheduler(cfg)

        def train_fn(vcfg, name: str):
            # 开启追踪时，为每个变体创建独立的嵌套 W&B run
            if track:
                finish_wandb()
                OmegaConf.set_struct(vcfg, False)
                vcfg.wandb.enabled = True
                vcfg.wandb.mode = cfg.wandb.mode
                vcfg.run_name = f"{cfg.experiment.name}-{name}"
                init_wandb(vcfg)
            return _dry_run(vcfg, name)

        results = scheduler.run(train_fn)
        logger.info("Ablation results: %s", results)
    finally:
        teardown_runtime()


if __name__ == "__main__":
    main()
