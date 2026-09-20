"""PostTrainLab 日志初始化：控制台 + Hydra 运行目录下的文件。"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any


def _hydra_output_dir(fallback: Path) -> Path:
    """优先返回 Hydra 本次运行的输出目录；未初始化时用 fallback。"""
    try:
        from hydra.core.hydra_config import HydraConfig

        if HydraConfig.initialized():
            return Path(HydraConfig.get().runtime.output_dir)
    except Exception:
        pass
    return fallback


def setup_logging(cfg: Any) -> logging.Logger:
    """
    根据 cfg.logging 配置根 logger 与 ``posttrainlab`` logger。

    - 控制台：打印到 stdout
    - 文件：写到 Hydra run 目录（若可用），否则写到 ``output_dir/logs``
    """
    log_cfg = cfg.logging
    # 日志级别：INFO / DEBUG / WARNING 等
    level_name = str(log_cfg.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = log_cfg.get("format", "%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    datefmt = log_cfg.get("datefmt", "%Y-%m-%d %H:%M:%S")
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    handlers: list[logging.Handler] = []
    # 是否输出到终端
    if log_cfg.get("to_console", True):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        handlers.append(sh)

    # 是否写入日志文件（默认 run.log）
    if log_cfg.get("to_file", True):
        out_dir = _hydra_output_dir(Path(cfg.output_dir) / "logs")
        out_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(out_dir / log_cfg.get("filename", "run.log"), encoding="utf-8")
        fh.setFormatter(formatter)
        handlers.append(fh)

    # 重置 root，避免重复 handler（多次调用 / Hydra 自带日志冲突）
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    for h in handlers:
        root.addHandler(h)

    # 降低 hydra / datasets 等库的刷屏日志
    for name in log_cfg.get("quiet_libs", []) or []:
        logging.getLogger(str(name)).setLevel(logging.WARNING)

    logger = logging.getLogger("posttrainlab")
    logger.setLevel(level)
    logger.debug("Logging initialized (level=%s)", level_name)
    return logger


def get_logger(name: str = "posttrainlab") -> logging.Logger:
    """按名称取 logger；业务代码可用 get_logger('posttrainlab.xxx')。"""
    return logging.getLogger(name)
