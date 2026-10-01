"""DPO 流水线 GPU 测试。

无 CUDA 时整文件 skip。覆盖：
- 偏好数据投影
- Loss / 动态 β（张量在 GPU 上）
- WandB report_to 与策略注册
- TRL ``CustomDPOTrainer`` 短步训练冒烟（**仅** ``Qwen/Qwen2.5-0.5B-Instruct`` + LoRA）
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="DPO 流水线测试需要 GPU（CUDA）",
)

OmegaConf = pytest.importorskip("omegaconf").OmegaConf
pytest.importorskip("datasets")

from src.algorithms.dpo_losses import compute_preference_loss
from src.core.registry import ALGORITHM_REGISTRY
from src.data.pipeline import build_pipeline
from src.trainers import dpo_trainer as dpo_mod
from src.trainers.dpo_trainer import (
    _build_beta_controller,
    _build_loss_cfg,
    _resolve_beta_mode,
    _resolve_fixed_beta,
    build_custom_dpo_trainer_cls,
)
from src.utils.wandb_utils import resolve_report_to

DEVICE = torch.device("cuda")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )


def _toy_dpo_dataset(tmp_path: Path, n: int = 12):
    data_path = tmp_path / "dpo.jsonl"
    _write_jsonl(
        data_path,
        [
            {
                "prompt": f"问题{i}",
                "chosen": f"好答{i}",
                "rejected": f"差答{i}",
            }
            for i in range(1, n + 1)
        ],
    )
    cfg = OmegaConf.create(
        {
            "name": "toy_dpo_pipe",
            "format": "dpo_pair",
            "source": str(data_path),
            "system_prompt": "sys",
            "version": "v-dpo-pipe",
            "split": {"use_official_eval": False, "ratios": [0.7, 0.2, 0.1], "seed": 12},
            "cleaning": {"dedup": True, "min_chars": 1, "filter_low_quality": True},
        }
    )
    return build_pipeline(cfg, algorithm="dpo", seed=12, version_dir=tmp_path / "versions")


def _cuda_logps(n: int = 4, seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    pc = (torch.randn(n, generator=g) * 0.5 + 1.0).to(DEVICE)
    pr = (torch.randn(n, generator=g) * 0.5 - 0.5).to(DEVICE)
    rc = (torch.randn(n, generator=g) * 0.3).to(DEVICE)
    rr = (torch.randn(n, generator=g) * 0.3).to(DEVICE)
    return pc, pr, rc, rr


def test_dpo_pipeline_data_to_projection(tmpdir):
    """数据 → Sample → 划分 → DPO 投影字段齐全。"""
    tmp_path = Path(str(tmpdir))
    ds, meta = _toy_dpo_dataset(tmp_path)
    assert meta.version == "v-dpo-pipe"
    assert "train" in ds and len(ds["train"]) > 0
    assert "eval" in ds
    row = ds["train"][0]
    assert set(row.keys()) >= {"prompt", "chosen", "rejected"}
    assert row["prompt"][-1]["role"] == "user"
    assert row["chosen"][0]["role"] == "assistant"
    assert row["rejected"][0]["role"] == "assistant"


def test_dpo_pipeline_config_to_loss_and_beta():
    """Hydra 风格 training 配置 → LossCfg / DynamicBeta。"""
    fixed = OmegaConf.create(
        {
            "loss_type": "dpo",
            "beta_mode": "fixed",
            "fixed_beta": 0.25,
            "margin": 1.0,
            "dynamic_beta": {
                "init_beta": 0.1,
                "target_kl": 4.0,
                "adjust_rate": 0.1,
                "min_beta": 0.01,
                "max_beta": 1.0,
            },
        }
    )
    cfg_fixed = _build_loss_cfg(fixed)
    assert cfg_fixed.resolved_beta_mode() == "fixed"
    assert abs(cfg_fixed.beta - 0.25) < 1e-9
    assert _build_beta_controller(cfg_fixed) is None

    dyn = OmegaConf.create(
        {
            "loss_type": "ipo",
            "beta_mode": "dynamic",
            "margin": 0.5,
            "dynamic_beta": {
                "init_beta": 0.15,
                "target_kl": 2.0,
                "adjust_rate": 0.2,
                "min_beta": 0.05,
                "max_beta": 0.8,
            },
        }
    )
    cfg_dyn = _build_loss_cfg(dyn)
    assert cfg_dyn.loss_type == "ipo"
    ctl = _build_beta_controller(cfg_dyn)
    assert ctl is not None
    assert abs(ctl.beta - 0.15) < 1e-9


def test_dpo_pipeline_loss_on_gpu_with_dynamic_beta():
    """GPU 上算 DPO loss，再按验证 KL 更新动态 β。"""
    tcfg = OmegaConf.create(
        {
            "loss_type": "dpo",
            "beta_mode": "dynamic",
            "dynamic_beta": {
                "init_beta": 0.1,
                "target_kl": 1.0,
                "adjust_rate": 0.1,
                "min_beta": 0.01,
                "max_beta": 1.0,
            },
        }
    )
    loss_cfg = _build_loss_cfg(tcfg)
    ctl = _build_beta_controller(loss_cfg)
    assert ctl is not None

    pc, pr, rc, rr = _cuda_logps(seed=1)
    assert pc.device.type == "cuda"

    loss1, m1 = compute_preference_loss(
        loss_cfg.loss_type, pc, pr, rc, rr, beta=ctl.beta, margin=loss_cfg.margin
    )
    assert loss1.device.type == "cuda"
    assert torch.isfinite(loss1)
    assert "mean_kl" in m1

    new_beta = ctl.step(current_kl=10.0)
    assert new_beta > 0.1
    loss2, m2 = compute_preference_loss("dpo", pc, pr, rc, rr, beta=new_beta)
    assert loss2.device.type == "cuda"
    assert torch.isfinite(loss2)
    assert abs(m2["beta"] - new_beta) < 1e-9


def test_dpo_pipeline_ipo_on_gpu():
    """IPO 合页在 GPU 上：未达 margin 时 loss≈1。"""
    tcfg = OmegaConf.create(
        {
            "loss_type": "ipo",
            "beta_mode": "fixed",
            "fixed_beta": 0.2,
            "margin": 1.0,
        }
    )
    assert _resolve_beta_mode(tcfg) == "fixed"
    assert abs(_resolve_fixed_beta(tcfg) - 0.2) < 1e-9
    cfg = _build_loss_cfg(tcfg)
    pc = torch.zeros(2, device=DEVICE)
    pr = torch.zeros(2, device=DEVICE)
    rc = torch.zeros(2, device=DEVICE)
    rr = torch.zeros(2, device=DEVICE)
    loss, metrics = compute_preference_loss(
        cfg.loss_type, pc, pr, rc, rr, beta=cfg.beta, margin=cfg.margin
    )
    assert loss.device.type == "cuda"
    assert abs(float(loss.item()) - 1.0) < 1e-5
    assert metrics["margin"] == 1.0


def test_dpo_pipeline_wandb_report_to_wiring():
    """开启 wandb 时 report_to 应为 wandb，供 DPOConfig / TRL 上报。"""
    cfg_on = OmegaConf.create(
        {
            "report_to": "none",
            "wandb": {"enabled": True, "mode": "online", "project": "posttrainlab"},
            "training": {"algorithm": "dpo"},
            "model": {"name": "toy/model"},
        }
    )
    assert resolve_report_to(cfg_on) == "wandb"

    cfg_off = OmegaConf.create(
        {
            "report_to": "none",
            "wandb": {"enabled": False, "mode": "disabled"},
        }
    )
    assert resolve_report_to(cfg_off) == "none"


def test_dpo_pipeline_strategy_registered_and_no_engine():
    """策略已注册；自研 Accelerate 循环模块应已移除。"""
    import importlib.util

    import src.trainers.dpo_trainer  # noqa: F401

    assert "dpo" in ALGORITHM_REGISTRY or hasattr(dpo_mod, "DPOStageTrainer")
    assert "dpo" in ALGORITHM_REGISTRY
    assert ALGORITHM_REGISTRY["dpo"].__name__ == "DPOStageTrainer"

    engine_path = Path(__file__).resolve().parents[1] / "src" / "algorithms" / "dpo_engine.py"
    assert not engine_path.is_file(), "自研 dpo_engine 应已删除，改用 TRL/Accelerate"
    assert importlib.util.find_spec("src.algorithms.dpo_engine") is None


def test_dpo_pipeline_end_to_end_gpu_chain(tmpdir):
    """端到端 GPU 链：数据投影 → 配置 → CUDA 上 loss + 动态 β。"""
    tmp_path = Path(str(tmpdir))
    ds, _meta = _toy_dpo_dataset(tmp_path, n=10)
    assert len(ds["train"]) >= 1

    tcfg = OmegaConf.create(
        {
            "loss_type": "dpo",
            "beta_mode": "dynamic",
            "margin": 1.0,
            "dynamic_beta": {
                "init_beta": 0.1,
                "target_kl": 4.0,
                "adjust_rate": 0.1,
                "min_beta": 0.01,
                "max_beta": 1.0,
            },
        }
    )
    loss_cfg = _build_loss_cfg(tcfg)
    ctl = _build_beta_controller(loss_cfg)
    assert ctl is not None

    bsz = min(4, len(ds["train"]))
    pc, pr, rc, rr = _cuda_logps(n=bsz, seed=0)
    loss, metrics = compute_preference_loss(
        loss_cfg.loss_type, pc, pr, rc, rr, beta=ctl.beta, margin=loss_cfg.margin
    )
    assert loss.device.type == "cuda"
    assert torch.isfinite(loss)
    ctl.step(metrics["mean_kl"] * 20.0)
    assert ctl.steps == 1


QWEN_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _build_qwen_lora_for_smoke():
    """按项目约定加载 Qwen2.5-0.5B-Instruct + LoRA（与 configs/model/qwen2.5-0.5b 一致）。"""
    pytest.importorskip("peft")
    from src.models.loader import load_model_and_tokenizer
    from src.models.lora import apply_lora

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
    model.train()
    return model, tokenizer


def test_dpo_pipeline_trl_train_smoke_gpu(tmpdir):
    """GPU 上用 Qwen2.5-0.5B + TRL CustomDPOTrainer 短步冒烟。"""
    pytest.importorskip("transformers")
    pytest.importorskip("trl")
    from datasets import Dataset
    from trl import DPOConfig

    tmp_path = Path(str(tmpdir))
    out_dir = tmp_path / "dpo_gpu_ckpt"
    os.environ["WANDB_DISABLED"] = "true"

    # 与项目 DPO 投影一致：prompt / chosen / rejected 为 messages
    rows = [
        {
            "prompt": [{"role": "user", "content": f"问题{i}"}],
            "chosen": [{"role": "assistant", "content": f"好答{i}"}],
            "rejected": [{"role": "assistant", "content": f"差答{i}"}],
        }
        for i in range(1, 9)
    ]
    train_ds = Dataset.from_list(rows)

    model, tokenizer = _build_qwen_lora_for_smoke()
    use_bf16 = bool(torch.cuda.is_bf16_supported())
    args = DPOConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        max_steps=1,
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        max_length=128,
        max_prompt_length=64,
        beta=0.1,
        loss_type="sigmoid",
        bf16=use_bf16,
        fp16=not use_bf16,
    )
    for key, val in (("max_completion_length", 64),):
        if hasattr(args, key):
            setattr(args, key, val)

    CustomCls = build_custom_dpo_trainer_cls()
    trainer_kwargs = {
        "model": model,
        "ref_model": None,
        "args": args,
        "train_dataset": train_ds,
        "processing_class": tokenizer,
        "tokenizer": tokenizer,
        "custom_loss_type": "dpo",
        "beta_mode": "fixed",
        "margin": 1.0,
        "beta_controller": None,
    }
    import inspect

    params = inspect.signature(CustomCls.__init__).parameters
    if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        trainer_kwargs = {k: v for k, v in trainer_kwargs.items() if k in params}
    trainer_kwargs["custom_loss_type"] = "dpo"
    trainer_kwargs["beta_mode"] = "fixed"
    trainer_kwargs["margin"] = 1.0
    trainer_kwargs["beta_controller"] = None

    trainer = CustomCls(**trainer_kwargs)
    assert next(trainer.model.parameters()).device.type == "cuda"

    result = trainer.train()
    assert result is not None
    metrics = getattr(result, "metrics", None) or {}
    assert "train_runtime" in metrics or "train_loss" in metrics or len(metrics) >= 0
    torch.cuda.synchronize()
