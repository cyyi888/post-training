"""Shared kwargs for HF/TRL TrainingArguments."""

from __future__ import annotations

from typing import Any

from src.utils.wandb_utils import default_run_name, resolve_report_to


def trainer_logging_kwargs(cfg: Any) -> dict[str, Any]:
    return {
        "report_to": resolve_report_to(cfg),
        "run_name": default_run_name(cfg),
        "logging_dir": f"{cfg.output_dir}/logs/{cfg.training.algorithm}",
    }
