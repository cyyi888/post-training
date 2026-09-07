#!/usr/bin/env python
"""One-click training entrypoint (Hydra)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Allow `python scripts/train.py` from repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Register trainers & datasets via import side-effects
import src.data.loaders  # noqa: F401
import src.trainers.dpo_trainer  # noqa: F401
import src.trainers.grpo_trainer  # noqa: F401
import src.trainers.sft_trainer  # noqa: F401
from src.core.registry import ALGORITHM_REGISTRY
from src.data.loaders import load_preference_dataset, load_task_dataset
from src.models.loader import load_model_and_tokenizer
from src.models.lora import apply_lora
from src.utils.seed import same_seeds


def _build_trainer(cfg: DictConfig):
    algo = cfg.training.algorithm
    if algo not in ALGORITHM_REGISTRY:
        raise KeyError(f"Unknown algorithm '{algo}'. Registered: {list(ALGORITHM_REGISTRY)}")

    model, tokenizer = load_model_and_tokenizer(
        cfg.model.name,
        use_gpu=cfg.use_gpu,
        torch_dtype=cfg.model.get("torch_dtype"),
        trust_remote_code=cfg.model.get("trust_remote_code", True),
    )
    if cfg.model.get("use_lora", False):
        model = apply_lora(model, cfg.model.lora)

    TrainerCls = ALGORITHM_REGISTRY[algo]

    if algo == "dpo":
        ds = load_preference_dataset(max_samples=cfg.data.get("max_train_samples", 256))
        return TrainerCls(cfg, model, tokenizer, ds)

    data = load_task_dataset(cfg.data)
    if algo == "sft":
        # SFT expects chat messages; gsm8k template yields prompt — for demo use train split as-is
        return TrainerCls(cfg, model, tokenizer, data["train"], eval_dataset=data.get("eval"))
    return TrainerCls(cfg, model, tokenizer, data["train"])


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    if cfg.get("hf_endpoint"):
        os.environ.setdefault("HF_ENDPOINT", cfg.hf_endpoint)
    same_seeds(cfg.seed)
    print(OmegaConf.to_yaml(cfg))

    trainer = _build_trainer(cfg)
    result = trainer.train()
    print("\n=== Train result ===")
    print(result.metrics)
    if result.checkpoint_dir:
        print(f"checkpoint: {result.checkpoint_dir}")


if __name__ == "__main__":
    main()
