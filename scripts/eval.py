#!/usr/bin/env python
"""One-click evaluation entrypoint (Hydra)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.data.loaders  # noqa: F401
from src.data.loaders import load_task_dataset
from src.evaluators.gsm8k_eval import GSM8KEvaluator
from src.evaluators.report import save_report
from src.models.loader import load_model_and_tokenizer
from src.utils.seed import same_seeds


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    if cfg.get("hf_endpoint"):
        os.environ.setdefault("HF_ENDPOINT", cfg.hf_endpoint)
    same_seeds(cfg.seed)

    model, tokenizer = load_model_and_tokenizer(
        cfg.model.name,
        use_gpu=cfg.use_gpu,
        torch_dtype=cfg.model.get("torch_dtype"),
        trust_remote_code=cfg.model.get("trust_remote_code", True),
    )
    data = load_task_dataset(cfg.data)
    eval_ds = data["eval"]

    task = cfg.get("task", cfg.data.name)
    if task == "gsm8k":
        evaluator = GSM8KEvaluator(cfg, model, tokenizer)
    else:
        raise NotImplementedError(f"Eval task '{task}' not implemented yet")

    metrics = evaluator.evaluate(eval_ds)
    print("\n=== Eval metrics ===")
    print(metrics)

    if cfg.get("save_report", True):
        path = Path(cfg.output_dir) / "logs" / cfg.get("report_name", "eval_report.json")
        save_report(metrics, path)
        print(f"report → {path}")


if __name__ == "__main__":
    main()
