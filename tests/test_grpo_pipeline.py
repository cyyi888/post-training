"""【设备：GPU / CUDA】GRPO 在线 RL 流水线冒烟。

原因：需真实 ``Qwen/Qwen2.5-0.5B-Instruct`` + LoRA 做 generate 与短步策略更新，
涉及大批量矩阵乘与 KV cache，必须在 GPU 上验证。无 CUDA 时整文件 skip。

轻量 Rollout 控制流单测见 ``tests/test_rollout.py``（CPU）。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GRPO 流水线冒烟需要 GPU（CUDA）",
)

OmegaConf = pytest.importorskip("omegaconf").OmegaConf
pytest.importorskip("transformers")
pytest.importorskip("peft")

from src.algorithms.grpo_engine import GRPOEngine
from src.algorithms.reward_functions import RewardEngine
from src.algorithms.rollout import GRPORolloutPipeline, RolloutConfig
from src.models.loader import load_model_and_tokenizer
from src.models.lora import apply_lora

QWEN_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _build_qwen_lora_for_smoke():
    """按项目约定加载 Qwen2.5-0.5B-Instruct + LoRA。"""
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


def test_grpo_rollout_pipeline_smoke_gpu():
    """GPU 上用 Qwen 做一轮短 Rollout。"""
    os.environ["WANDB_DISABLED"] = "true"
    model, tokenizer = _build_qwen_lora_for_smoke()
    model.eval()

    reward = RewardEngine(
        ["format", "length"],
        weights=[1.0, 0.1],
        length_reward={"min_tokens": 1, "max_tokens": 128, "target_min": 1, "target_max": 128},
    )
    cfg = RolloutConfig(
        num_generations=2,
        max_completion_length=24,
        temperature=0.8,
        top_p=0.95,
        top_k=0,
        max_resample=0,
        max_group_resample=0,
        min_valid_per_group=1,
        require_answer_format=False,
        filter_extreme_length=False,
        clear_cache=True,
        bucket=False,
        fix_generation_seed=True,
        generation_seed=0,
    )
    pipe = GRPORolloutPipeline(reward, cfg)
    rows = [
        {
            "prompt": [{"role": "user", "content": "1+1=? Reply briefly."}],
            "ground_truth": "2",
        }
    ]
    batch = pipe.run(rows, model, tokenizer, step=0)
    assert batch is not None
    assert not batch.empty
    assert batch.input_ids.device.type == "cuda"
    assert batch.rewards.device.type == "cuda"
    assert "perf/tokens_per_sec" in batch.metrics
    torch.cuda.synchronize()


def test_grpo_engine_train_loop_smoke_gpu(tmpdir):
    """GPU 上 GRPOEngine 短步更新冒烟。"""
    os.environ["WANDB_DISABLED"] = "true"
    tmp_path = Path(str(tmpdir))
    model, tokenizer = _build_qwen_lora_for_smoke()

    reward = RewardEngine(
        ["accuracy", "format", "length"],
        weights=[1.0, 0.2, 0.1],
        length_reward={"min_tokens": 1, "max_tokens": 128, "target_min": 1, "target_max": 128},
    )
    rollout_cfg = RolloutConfig(
        num_generations=2,
        max_completion_length=24,
        temperature=0.8,
        top_p=0.95,
        max_resample=0,
        max_group_resample=0,
        min_valid_per_group=1,
        require_answer_format=False,
        filter_extreme_length=False,
        clear_cache=True,
        bucket=True,
        bucket_boundaries=[128, 256, 512],
        fix_generation_seed=True,
        generation_seed=1,
    )
    engine = GRPOEngine(
        model,
        tokenizer,
        reward,
        num_generations=2,
        max_completion_length=24,
        temperature=0.8,
        clip_range=0.2,
        beta=0.0,
        num_iterations=1,
        mixed_precision="bf16" if torch.cuda.is_bf16_supported() else "fp16",
        rollout_config=rollout_cfg,
        seed=0,
    )
    rows = [
        {
            "prompt": [{"role": "user", "content": "What is 2+2? Brief answer."}],
            "ground_truth": "4",
        },
        {
            "prompt": [{"role": "user", "content": "What is 3+1? Brief answer."}],
            "ground_truth": "4",
        },
    ]
    metrics = engine.train_loop(
        rows,
        epochs=1,
        batch_size=1,
        grad_accum=1,
        max_steps=1,
        logging_steps=1,
    )
    assert metrics.get("train/steps", 0) >= 1.0
    assert next(model.parameters()).device.type == "cuda"
    out = tmp_path / "grpo_gpu_smoke"
    out.mkdir(parents=True, exist_ok=True)
    (out / "ok.txt").write_text("1", encoding="utf-8")
    torch.cuda.synchronize()
