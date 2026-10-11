#!/usr/bin/env python
"""One-click evaluation entrypoint (Hydra) — 统一 ``evaluate(config, checkpoint)``。"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.data.loaders  # noqa: F401
import src.evaluators  # noqa: F401 — 注册 gsm8k / humaneval
from src.evaluators import evaluate
from src.evaluators.common import eval_section
from src.utils.runtime import prepare_runtime, teardown_runtime
from src.utils.wandb_utils import log_metrics

logger = logging.getLogger("posttrainlab.eval")


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    prepare_runtime(cfg, init_wb=bool(cfg.wandb.get("enabled", False)))
    try:
        ecfg = eval_section(cfg)
        ckpt = ecfg.get("checkpoint")
        result = evaluate(cfg, checkpoint=ckpt)
        logger.info(
            "Eval task=%s kind=%s ckpt=%s metrics=%s status=%s",
            result.task,
            result.checkpoint_kind,
            result.checkpoint,
            result.metrics,
            result.validation_status,
        )
        log_metrics({f"eval/{k}": v for k, v in result.metrics.items()})
        if result.report_path:
            logger.info("Report saved → %s", result.report_path)
    finally:
        teardown_runtime()


if __name__ == "__main__":
    main()
