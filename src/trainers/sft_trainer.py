"""SFT 训练策略：TRL SFTTrainer 封装，支持 Full / LoRA。

自研重点（见 ``sft_trl`` / ``sft_samplers`` / ``grad_clip``）：
自定义采样器、长度分组采样、梯度裁剪策略。
"""

from __future__ import annotations

import logging
from typing import Any

from src.core.base import BaseTrainer, TrainContext, TrainResult
from src.core.registry import ALGORITHM_REGISTRY, register

logger = logging.getLogger("posttrainlab.sft")


def _as_dict(section: Any) -> dict[str, Any]:
    if section is None:
        return {}
    if isinstance(section, dict):
        return dict(section)
    if hasattr(section, "items"):
        try:
            return dict(section.items())
        except Exception:
            pass
    data = getattr(section, "__dict__", None)
    if isinstance(data, dict):
        return {k: v for k, v in data.items() if not str(k).startswith("_")}
    return {}


def _log_trainable(model: Any) -> dict[str, float]:
    trainable = 0
    total = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    ratio = float(trainable) / float(total) if total else 0.0
    logger.info(
        "可训练参数 %s / %s (%.4f%%) | peft=%s",
        trainable,
        total,
        100.0 * ratio,
        getattr(model, "peft_config", None) is not None,
    )
    return {
        "model/trainable_params": float(trainable),
        "model/total_params": float(total),
        "model/trainable_ratio": ratio,
        "model/lora": 1.0 if getattr(model, "peft_config", None) is not None else 0.0,
    }


@register(ALGORITHM_REGISTRY, "sft")
class SFTStageTrainer(BaseTrainer):
    """监督微调：Full 或 LoRA；采样与裁剪为自研，其余走 TRL。"""

    name = "sft"

    def run_algorithm(self, ctx: TrainContext) -> TrainResult:
        from trl import SFTConfig

        from src.trainers.sft_trl import build_posttrain_sft_trainer

        train_ds, eval_ds = ctx.train_dataset, ctx.eval_dataset
        tcfg = ctx.config.training
        runtime = ctx.runtime
        out_dir = ctx.checkpoint_dir
        sampler_cfg = _as_dict(tcfg.get("sampler"))
        grad_clip_cfg = _as_dict(tcfg.get("grad_clip"))
        if not sampler_cfg:
            sampler_cfg = {"mode": "length_group", "mega_batch_mult": 8}
        if sampler_cfg.get("seed") in (None, "null"):
            sampler_cfg["seed"] = int(ctx.config.get("seed", 0))
        if not grad_clip_cfg:
            grad_clip_cfg = {"strategy": "norm", "max_norm": runtime.max_grad_norm}

        # 自研裁剪时关闭 HF 内置裁剪，避免双重缩放
        strategy = str(grad_clip_cfg.get("strategy", "norm")).lower()
        hf_args = dict(ctx.hf_args)
        if strategy not in ("hf", "default"):
            hf_args["max_grad_norm"] = 0.0

        max_steps = tcfg.get("max_steps", None)
        sft_kwargs: dict[str, Any] = {
            "output_dir": out_dir,
            "num_train_epochs": tcfg.num_train_epochs,
            "per_device_train_batch_size": tcfg.per_device_train_batch_size,
            "per_device_eval_batch_size": tcfg.get("per_device_eval_batch_size", 1),
            "logging_steps": tcfg.logging_steps,
            "eval_strategy": tcfg.get("eval_strategy", "no") if eval_ds is not None else "no",
            "save_strategy": tcfg.get("save_strategy", "epoch"),
            "max_length": tcfg.max_length,
            "assistant_only_loss": tcfg.get("assistant_only_loss", True),
            **hf_args,
        }
        if max_steps not in (None, "null"):
            sft_kwargs["max_steps"] = int(max_steps)

        # 长度分组时关闭 HF 自带 group_by_length，改走自研 sampler
        if str(sampler_cfg.get("mode", "")).lower() in ("length_group", "length", "bucket"):
            sft_kwargs["group_by_length"] = False

        args = SFTConfig(**sft_kwargs)
        param_metrics = _log_trainable(ctx.model)
        finetune = "lora" if param_metrics["model/lora"] > 0 else "full"
        logger.info(
            "SFT 基线启动 finetune=%s sampler=%s grad_clip=%s",
            finetune,
            sampler_cfg.get("mode"),
            strategy,
        )

        trainer = build_posttrain_sft_trainer(
            model=ctx.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=ctx.tokenizer,
            sampler_config=sampler_cfg,
            grad_clip_raw=grad_clip_cfg,
            default_max_norm=float(
                grad_clip_cfg.get("max_norm")
                if grad_clip_cfg.get("max_norm") not in (None, "null")
                else runtime.max_grad_norm
            ),
        )
        train_out = trainer.train(resume_from_checkpoint=ctx.resume_from)
        metrics = dict(train_out.metrics) if train_out and train_out.metrics else {}
        metrics.update(param_metrics)
        metrics["sft/finetune_lora"] = param_metrics["model/lora"]
        if eval_ds is not None and tcfg.get("eval_strategy", "no") != "no":
            metrics.update(trainer.evaluate())
        self.save(trainer.model, ctx.tokenizer, out_dir)
        return TrainResult(
            metrics=metrics,
            checkpoint_dir=out_dir,
            extra={"finetune": finetune, "sampler": sampler_cfg, "grad_clip": grad_clip_cfg},
        )
