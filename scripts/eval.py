#!/usr/bin/env python
"""One-click evaluation entrypoint (Hydra)."""

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
from src.evaluators.gsm8k_eval import GSM8KEvaluator
from src.evaluators.report import save_report
from src.models.loader import load_model_and_tokenizer
from src.utils.runtime import prepare_runtime, teardown_runtime
from src.utils.wandb_utils import log_metrics

logger = logging.getLogger("posttrainlab.eval")


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    # Eval does not need training.* for wandb naming; keep struct flexible
    OmegaConf.set_struct(cfg, False)
    prepare_runtime(cfg, init_wb=bool(cfg.wandb.get("enabled", False)))
    try:
        model, tokenizer = load_model_and_tokenizer(
            cfg.model.name,
            use_gpu=cfg.use_gpu,
            torch_dtype=cfg.model.get("torch_dtype"),
            trust_remote_code=cfg.model.get("trust_remote_code", True),
            chat_cfg=cfg.get("chat_template"),
        )
        from src.data.pipeline import build_pipeline

        data, version = build_pipeline(
            cfg.data,
            algorithm="grpo",
            seed=int(cfg.seed),
            version_dir=Path(cfg.output_dir) / "data_versions" / str(cfg.data.get("name", "data")),
        )
        cfg.data_version = version.to_dict()
        eval_ds = data["eval"] if "eval" in data else data["test"]
        task = cfg.get("task", cfg.data.name)
        if task == "gsm8k":
            evaluator = GSM8KEvaluator(cfg, model, tokenizer)
        else:
            raise NotImplementedError(f"Eval task '{task}' not implemented yet")

        metrics = evaluator.evaluate(eval_ds)
        logger.info("Eval metrics: %s", metrics)
        log_metrics({f"eval/{k}": v for k, v in metrics.items()})

        if cfg.get("save_report", True):
            from hydra.core.hydra_config import HydraConfig

            if HydraConfig.initialized():
                path = Path(HydraConfig.get().runtime.output_dir) / cfg.get(
                    "report_name", "eval_report.json"
                )
            else:
                path = Path(cfg.output_dir) / "logs" / cfg.get("report_name", "eval_report.json")
            save_report(metrics, path)
            logger.info("Report saved → %s", path)
    finally:
        teardown_runtime()


if __name__ == "__main__":
    main()
