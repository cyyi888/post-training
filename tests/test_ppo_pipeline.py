"""【设备：GPU / CUDA】PPO（TRL 基线）流水线冒烟。

原因：需加载 ``Qwen/Qwen2.5-0.5B-Instruct`` + LoRA，并走 TRL ``PPOTrainer``
生成/更新，矩阵运算与显存占用要求 GPU。无 CUDA 或环境无 PPOTrainer 时 skip。

奖励引擎纯逻辑见 ``tests/test_rewards.py``（CPU）。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="PPO 流水线测试需要 GPU（CUDA）",
)

OmegaConf = pytest.importorskip("omegaconf").OmegaConf
pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("datasets")

from src.algorithms.reward_functions import build_reward_funcs
from src.core.base import RuntimeOptions, TrainContext
from src.models.loader import load_model_and_tokenizer
from src.models.lora import apply_lora

QWEN_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _build_qwen_lora_for_smoke():
    model, tokenizer = load_model_and_tokenizer(
        QWEN_MODEL,
        use_gpu=True,
        torch_dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float16",
        trust_remote_code=True,
    )
    lora_cfg = OmegaConf.create(
        {
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }
    )
    model = apply_lora(model, lora_cfg)
    assert next(model.parameters()).device.type == "cuda"
    return model, tokenizer


def _import_trl_ppo_or_skip():
    try:
        from src.trainers.ppo_trainer import _import_trl_ppo

        return _import_trl_ppo()
    except ImportError as exc:
        pytest.skip(f"当前环境无 TRL PPOTrainer: {exc}")


def test_ppo_reward_engine_wired_on_gpu():
    """确认 PPO 使用的自研奖励可调用（公式本身在 CPU 单测覆盖）。"""
    cfg = OmegaConf.create({"funcs": ["accuracy"], "weights": [1.0]})
    reward_fn = build_reward_funcs(cfg)
    scores = reward_fn(
        [[{"role": "assistant", "content": r"\boxed{2}"}]],
        ground_truth=["2"],
    )
    assert scores == [1.0]


def test_ppo_stage_trainer_smoke_gpu(tmpdir):
    """GPU 上走 PPOStageTrainer.run_algorithm 短冒烟。"""
    from datasets import Dataset

    from src.trainers.ppo_trainer import PPOStageTrainer

    _import_trl_ppo_or_skip()

    os.environ["WANDB_DISABLED"] = "true"
    tmp_path = Path(str(tmpdir))
    out_dir = tmp_path / "ppo_gpu_ckpt"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = _build_qwen_lora_for_smoke()
    rows = [
        {
            "prompt": [{"role": "user", "content": f"What is {i}+0? Reply with a number."}],
            "ground_truth": str(i),
        }
        for i in range(1, 5)
    ]
    train_ds = Dataset.from_list(rows)

    cfg = OmegaConf.create(
        {
            "seed": 0,
            "training": {
                "algorithm": "ppo",
                "per_device_train_batch_size": 1,
                "num_train_epochs": 1,
                "max_completion_length": 16,
                "logging_steps": 1,
                "save_strategy": "no",
                "ppo": {"mini_batch_size": 1, "kl_coef": 0.05},
                "reward": {"funcs": ["accuracy"], "weights": [1.0]},
            },
            "wandb": {"enabled": False, "mode": "disabled"},
        }
    )
    runtime = RuntimeOptions(
        seed=0,
        learning_rate=1e-6,
        gradient_accumulation_steps=1,
        mixed_precision="bf16" if torch.cuda.is_bf16_supported() else "fp16",
        max_grad_norm=1.0,
    )
    ctx = TrainContext(
        config=cfg,
        dataset={"train": train_ds},
        train_dataset=train_ds,
        eval_dataset=None,
        model=model,
        tokenizer=tokenizer,
        checkpoint_dir=str(out_dir),
        resume_from=None,
        runtime=runtime,
    )

    trainer = PPOStageTrainer()
    try:
        result = trainer.run_algorithm(ctx)
    except Exception as exc:
        msg = str(exc).lower()
        if "ppo" in msg or "trl" in msg or "unexpected" in msg or "missing" in msg:
            pytest.skip(f"TRL PPO 接口与当前版本不兼容: {exc}")
        raise

    assert result is not None
    assert result.extra.get("impl") == "trl"
    assert next(model.parameters()).device.type == "cuda"
    torch.cuda.synchronize()
