#!/usr/bin/env python
"""Ablation experiment runner (Hydra)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment.scheduler import AblationScheduler


def _dry_run(cfg, variant_name: str) -> dict:
    """Placeholder train hook — records config fingerprint without GPU work."""
    t = cfg.get("training", {})
    return {
        "loss_type": str(t.get("loss_type", "")),
        "beta": float(t.get("beta", 0.0) or 0.0),
        "dynamic_beta": bool(t.get("dynamic_beta", False)),
        "penalty_enabled": bool(t.get("penalty", {}).get("enabled", False))
        if t.get("penalty")
        else False,
        "status": "dry_run",
    }


@hydra.main(version_base=None, config_path="../configs", config_name="experiment/ablation")
def main(cfg: DictConfig) -> None:
    if cfg.get("hf_endpoint"):
        os.environ.setdefault("HF_ENDPOINT", cfg.hf_endpoint)
    print(OmegaConf.to_yaml(cfg))
    scheduler = AblationScheduler(cfg)
    # Swap `_dry_run` for a real train callback when ready:
    #   from scripts.train import _build_trainer
    #   def train_fn(vcfg, name): return _build_trainer(vcfg).train().metrics
    results = scheduler.run(_dry_run)
    print("\n=== Ablation results ===")
    for row in results:
        print(row)


if __name__ == "__main__":
    main()
