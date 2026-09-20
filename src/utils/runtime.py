"""Shared runtime bootstrap for Hydra entrypoints."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.utils.env import load_dotenv
from src.utils.logging_utils import setup_logging
from src.utils.seed import same_seeds
from src.utils.wandb_utils import finish_wandb, init_wandb, resolve_report_to

logger = logging.getLogger("posttrainlab")


def prepare_runtime(cfg: DictConfig, *, init_wb: bool = True) -> Any:
    """
    加载 .env、固定种子、初始化日志与可选 W&B，并落盘 resolved 配置。

    返回 wandb Run（未启用则为 None）。
    """
    env_path = load_dotenv()

    if cfg.get("hf_endpoint"):
        os.environ.setdefault("HF_ENDPOINT", str(cfg.hf_endpoint))

    same_seeds(int(cfg.seed))
    setup_logging(cfg)
    if env_path is not None:
        logger.info("Loaded environment from %s", env_path)

    # Keep TRL report_to in sync with wandb group
    report_to = resolve_report_to(cfg)
    OmegaConf.set_struct(cfg, False)
    cfg.report_to = report_to

    try:
        from hydra.core.hydra_config import HydraConfig

        if HydraConfig.initialized():
            out = Path(HydraConfig.get().runtime.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(cfg, out / "resolved_config.yaml")
            logger.info("Hydra output dir: %s", out)
    except Exception:
        fallback = Path(cfg.output_dir) / "logs"
        fallback.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, fallback / "resolved_config.yaml")

    logger.info(
        "Runtime ready | algo=%s model=%s report_to=%s seed=%s",
        cfg.training.algorithm,
        cfg.model.name,
        cfg.report_to,
        cfg.seed,
    )
    logger.debug("Full config:\n%s", OmegaConf.to_yaml(cfg))

    run = init_wandb(cfg) if init_wb else None
    return run


def teardown_runtime() -> None:
    finish_wandb()
