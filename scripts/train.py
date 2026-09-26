#!/usr/bin/env python
"""一键训练入口：策略模式选取 trainer，统一 train(config, dataset)。"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 注册策略：SFT / DPO / GRPO / PPO
import src.trainers.dpo_trainer  # noqa: F401
import src.trainers.grpo_trainer  # noqa: F401
import src.trainers.ppo_trainer  # noqa: F401
import src.trainers.sft_trainer  # noqa: F401
from src.data.pipeline import build_pipeline
from src.trainers.factory import get_trainer


def run_training(cfg: DictConfig):
    """准备数据 → 按 algorithm 取策略 → trainer.train(config, dataset)。

    日志、W&B、断点续训和分布式由 BaseTrainer.train 统一处理。
    """
    import os

    if cfg.get("hf_endpoint"):
        os.environ.setdefault("HF_ENDPOINT", str(cfg.hf_endpoint))

    algo = cfg.training.algorithm
    version_dir = Path(cfg.output_dir) / "data_versions" / str(cfg.data.get("name", "data"))
    # PPO 暂无专用投影时复用 grpo 字段（prompt + ground_truth）
    project_algo = "grpo" if algo == "ppo" else algo
    ds, version = build_pipeline(
        cfg.data,
        algorithm=project_algo,
        seed=int(cfg.seed),
        version_dir=version_dir,
    )
    OmegaConf.set_struct(cfg, False)
    cfg.data_version = version.to_dict()

    trainer = get_trainer(algo)
    return trainer.train(cfg, ds)


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    main()
