#!/usr/bin/env python
"""One-click training entrypoint (Hydra)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.trainers.dpo_trainer  # noqa: F401
import src.trainers.grpo_trainer  # noqa: F401
import src.trainers.sft_trainer  # noqa: F401
from src.core.registry import ALGORITHM_REGISTRY
from src.data.pipeline import build_pipeline
from src.models.loader import load_model_and_tokenizer
from src.models.lora import apply_lora
from src.utils.runtime import prepare_runtime, teardown_runtime
from src.utils.wandb_utils import log_metrics

logger = logging.getLogger("posttrainlab.train")


def _build_trainer(cfg: DictConfig):
    algo = cfg.training.algorithm
    if algo not in ALGORITHM_REGISTRY:
        raise KeyError(f"Unknown algorithm '{algo}'. Registered: {list(ALGORITHM_REGISTRY)}")

    model, tokenizer = load_model_and_tokenizer(
        cfg.model.name,
        use_gpu=cfg.use_gpu,
        torch_dtype=cfg.model.get("torch_dtype"),
        trust_remote_code=cfg.model.get("trust_remote_code", True),
        chat_cfg=cfg.get("chat_template"),
    )
    if cfg.model.get("use_lora", False):
        model = apply_lora(model, cfg.model.lora)

    version_dir = (
        Path(cfg.output_dir) / "data_versions" / str(cfg.data.get("name", "data"))
    )
    ds, version = build_pipeline(
        cfg.data,
        algorithm=algo,
        seed=int(cfg.seed),
        version_dir=version_dir,
    )
    OmegaConf.set_struct(cfg, False)
    cfg.data_version = version.to_dict()
    logger.info(
        "绑定数据版本 version=%s hash=%s",
        version.version,
        version.hash[:12],
    )
    log_metrics(
        {
            "data/version": version.version,
            "data/hash": version.short_hash(),
            "data/num_samples": version.num_samples,
            "data/train_size": version.split_sizes.get("train", 0),
            "data/val_size": version.split_sizes.get("val", 0),
            "data/test_size": version.split_sizes.get("test", 0),
        }
    )

    TrainerCls = ALGORITHM_REGISTRY[algo]
    train_ds = ds["train"]
    eval_ds = ds["eval"] if "eval" in ds else None
    if algo == "sft":
        return TrainerCls(cfg, model, tokenizer, train_ds, eval_dataset=eval_ds)
    return TrainerCls(cfg, model, tokenizer, train_ds)


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    prepare_runtime(cfg, init_wb=True)
    try:
        trainer = _build_trainer(cfg)
        result = trainer.train()
        logger.info("Train metrics: %s", result.metrics)
        log_metrics({f"final/{k}": v for k, v in result.metrics.items()})
        if result.checkpoint_dir:
            logger.info("Checkpoint: %s", result.checkpoint_dir)
    finally:
        teardown_runtime()


if __name__ == "__main__":
    main()
