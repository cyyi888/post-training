"""Weights & Biases（W&B）相关工具：初始化、打点、结束 run。"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger("posttrainlab.wandb")


def default_run_name(cfg: Any) -> str:
    """
    生成本次实验的 run 显示名。

    优先级：顶层 run_name > wandb.run_name > 「算法名-模型短名」。
    """
    if cfg.get("run_name"):
        return str(cfg.run_name)
    w = cfg.get("wandb", {})
    if w is not None and w.get("run_name"):
        return str(w.run_name)
    model_short = str(cfg.model.name).split("/")[-1]
    algo = str(cfg.training.algorithm)
    return f"{algo}-{model_short}"


def resolve_report_to(cfg: Any) -> str:
    """
    把 wandb 配置映射成 HuggingFace / TRL 的 ``report_to``。

    wandb 开启且 mode 不是 disabled 时返回 ``"wandb"``，否则用配置里的 report_to。
    """
    w = cfg.get("wandb", {})
    if w is not None and w.get("enabled") and str(w.get("mode", "online")) != "disabled":
        return "wandb"
    report = cfg.get("report_to", "none")
    if report is None or report == "" or report == ["none"]:
        return "none"
    if isinstance(report, (list, tuple)):
        return report[0] if report else "none"
    return str(report)


def init_wandb(cfg: DictConfig | Any):
    """
    在 ``wandb.enabled`` 为真时初始化一次 W&B run。

    返回 wandb Run；未启用时返回 None，并设置 WANDB_DISABLED。
    """
    w = cfg.get("wandb", None)
    # 未配置 / 未启用 / mode=disabled → 直接跳过
    if w is None or not w.get("enabled") or str(w.get("mode", "disabled")) == "disabled":
        os.environ.setdefault("WANDB_DISABLED", "true")
        logger.info("W&B disabled")
        return None

    try:
        import wandb
    except ImportError as e:
        raise ImportError("已开启 wandb 但未安装，请执行: pip install wandb") from e

    # 清掉禁用标记，并准备本地缓存目录（默认 outputs/wandb）
    os.environ.pop("WANDB_DISABLED", None)
    os.environ.setdefault("WANDB_DIR", str(w.dir))
    Path(str(w.dir)).mkdir(parents=True, exist_ok=True)

    run_name = default_run_name(cfg)
    tags = list(w.tags) if w.get("tags") else None
    # 把整份 Hydra 配置写入本次 run，便于网页端复现
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    run = wandb.init(
        project=str(w.project),
        entity=w.get("entity"),
        name=run_name,
        config=config_dict,
        dir=str(w.dir),
        mode=str(w.mode),  # online | offline | disabled
        tags=tags,
        notes=w.get("notes"),
        reinit=True,  # 允许同一进程内再次 init（如消融多变体）
    )
    logger.info("W&B run started: project=%s name=%s mode=%s", w.project, run_name, w.mode)
    return run


def log_metrics(metrics: dict[str, Any], step: int | None = None) -> None:
    """向当前 W&B run 写入指标；无活跃 run 时静默跳过。"""
    try:
        import wandb

        if wandb.run is None:
            return
        # 只上传可序列化的标量/字符串，避免复杂对象报错
        payload = {k: v for k, v in metrics.items() if isinstance(v, (int, float, bool, str))}
        if not payload:
            return
        if step is None:
            wandb.log(payload)
        else:
            wandb.log(payload, step=step)
    except Exception as e:
        logger.warning("Failed to log metrics to W&B: %s", e)


def finish_wandb() -> None:
    """结束当前 W&B run，刷新并上传本地缓冲。"""
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
            logger.info("W&B run finished")
    except Exception as e:
        logger.warning("Failed to finish W&B run: %s", e)
