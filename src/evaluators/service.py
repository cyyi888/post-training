"""统一 Checkpoint 评测入口：``evaluate(config, checkpoint)``。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from src.evaluators.checkpoint import (
    CheckpointError,
    load_checkpoint_for_eval,
    resolve_checkpoint,
)
from src.evaluators.common import eval_section
from src.evaluators.factory import build_evaluator
from src.evaluators.result import EvaluationResult

logger = logging.getLogger("posttrainlab.eval")

BuildDatasetFn = Callable[[Any, str], Any]
BuildEvaluatorFn = Callable[[str, Any, Any, Any], Any]


def resolve_eval_task(config: Any) -> str:
    """配置化选择 GSM8K / HumanEval（及别名）。"""
    ecfg = eval_section(config)
    task = ecfg.get("task")
    if not task:
        data = config.get("data") if hasattr(config, "get") else getattr(config, "data", None)
        if data is not None:
            task = data.get("name") if hasattr(data, "get") else getattr(data, "name", None)
    task = str(task or "gsm8k").lower().strip()
    aliases = {
        "openai_humaneval": "humaneval",
        "human_eval": "humaneval",
        "gsm-8k": "gsm8k",
    }
    return aliases.get(task, task)


def evaluate(
    config: Any,
    checkpoint: str | Path | None = None,
    *,
    dataset: Any | None = None,
    load_model_fn: Callable | None = None,
    load_adapter_fn: Callable | None = None,
    build_dataset_fn: BuildDatasetFn | None = None,
    build_evaluator_fn: BuildEvaluatorFn | None = None,
    report_dir: str | Path | None = None,
    save_report_flag: bool | None = None,
) -> EvaluationResult:
    """
    统一评测接口。

    1. 解析 checkpoint（显式路径 / 阶段别名 / latest）
    2. 按 Full / LoRA 加载模型与 tokenizer（生产加载器，可注入）
    3. 按 ``eval.task`` 构造既有 Evaluator 并跑 ``evaluate(dataset)``
    4. 写出 ``eval_report.json``（或配置中的 report_name）

    真实权重正确性标记为 ``GPU_PENDING``；本地可用注入 Mock 做 CPU 单测。
    """
    ckpt_path = resolve_checkpoint(config, checkpoint)
    model, tokenizer, kind = load_checkpoint_for_eval(
        config,
        ckpt_path,
        load_model_fn=load_model_fn,
        load_adapter_fn=load_adapter_fn,
    )

    task = resolve_eval_task(config)
    eval_ds = dataset
    if eval_ds is None:
        builder = build_dataset_fn or _default_build_dataset
        eval_ds = builder(config, task)

    maker = build_evaluator_fn or build_evaluator
    evaluator = maker(task, config, model, tokenizer)
    metrics = evaluator.evaluate(eval_ds)
    # 确保值为 float，便于 JSON / 汇总
    metrics_f = {
        str(k): float(v) if isinstance(v, (int, float)) else v
        for k, v in dict(metrics).items()
    }

    result = EvaluationResult(
        task=task,
        checkpoint=str(ckpt_path),
        checkpoint_kind=kind,
        metrics={k: float(v) for k, v in metrics_f.items() if isinstance(v, (int, float))},
        extra={
            "raw_metrics": metrics_f,
            "model_name": _model_name(config),
        },
        validation_status="GPU_PENDING",
    )

    ecfg = eval_section(config)
    do_save = save_report_flag
    if do_save is None:
        do_save = bool(ecfg.get("save_report", True))
    if do_save:
        report_name = str(ecfg.get("report_name") or "eval_report.json")
        out_dir = Path(report_dir) if report_dir is not None else _default_report_dir(config)
        result.save_report(out_dir / report_name)
        logger.info("Eval report → %s", result.report_path)

    logger.info(
        "Eval done task=%s kind=%s ckpt=%s metrics=%s",
        task,
        kind,
        ckpt_path,
        result.metrics,
    )
    return result


def _default_build_dataset(config: Any, task: str) -> Any:
    from src.data.pipeline import build_pipeline

    data_cfg = config.get("data") if hasattr(config, "get") else config.data
    seed = int(config.get("seed", 0) if hasattr(config, "get") else getattr(config, "seed", 0))
    out = Path(str(config.get("output_dir", "outputs") if hasattr(config, "get") else "outputs"))
    name = "data"
    if data_cfg is not None and hasattr(data_cfg, "get"):
        name = str(data_cfg.get("name", "data"))
    data, version = build_pipeline(
        data_cfg,
        algorithm="grpo",
        seed=seed,
        version_dir=out / "data_versions" / name,
    )
    if hasattr(config, "__setattr__"):
        try:
            from omegaconf import OmegaConf

            OmegaConf.set_struct(config, False)
            config.data_version = version.to_dict()
        except Exception:
            pass
    if hasattr(data, "keys"):
        if "eval" in data:
            return data["eval"]
        if "test" in data:
            return data["test"]
    return data


def _default_report_dir(config: Any) -> Path:
    try:
        from hydra.core.hydra_config import HydraConfig

        if HydraConfig.initialized():
            return Path(HydraConfig.get().runtime.output_dir)
    except Exception:
        pass
    out = Path(str(config.get("output_dir", "outputs") if hasattr(config, "get") else "outputs"))
    return out / "logs" / "eval"


def _model_name(config: Any) -> str:
    model = config.get("model") if hasattr(config, "get") else getattr(config, "model", None)
    if model is None:
        return ""
    if hasattr(model, "get"):
        return str(model.get("name", ""))
    return str(getattr(model, "name", ""))


__all__ = [
    "evaluate",
    "resolve_eval_task",
]
