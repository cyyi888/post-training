"""PPO 基线：仅封装 TRL 的 PPOTrainer，用于和自研 GRPO 对比。"""

from __future__ import annotations

import inspect
import logging
from typing import Any

from src.algorithms.reward_functions import build_reward_funcs
from src.core.base import BaseTrainer, TrainContext, TrainResult
from src.core.registry import ALGORITHM_REGISTRY, register
from src.data.chat_templates import render_chat

logger = logging.getLogger("posttrainlab.ppo")


def _import_trl_ppo():
    """兼容不同 TRL 版本的 PPO 导出位置。"""
    try:
        from trl import PPOConfig, PPOTrainer

        return PPOConfig, PPOTrainer
    except ImportError:
        pass
    try:
        from trl.experimental.ppo import PPOConfig, PPOTrainer

        return PPOConfig, PPOTrainer
    except ImportError as exc:
        raise ImportError(
            "PPO 基线需要带 PPOTrainer 的 trl。"
            "当前环境未导出 PPO（新版 TRL 可能已移除）。请安装含 PPO 的 trl，或改用自研 training=grpo。"
        ) from exc


def _filter_kwargs(fn: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    params = inspect.signature(fn).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


@register(ALGORITHM_REGISTRY, "ppo")
class PPOStageTrainer(BaseTrainer):
    """
    基于 TRL 的 PPO 基线封装。

    不自研 PPO 损失；只负责组配置、奖励函数和调用 ``PPOTrainer``，
    用来和自研 GRPO 做对照实验。
    """

    name = "ppo"

    def run_algorithm(self, ctx: TrainContext) -> TrainResult:
        train_ds = ctx.train_dataset
        model, tokenizer = ctx.model, ctx.tokenizer
        PPOConfig, PPOTrainer = _import_trl_ppo()
        tcfg = ctx.config.training
        out_dir = ctx.checkpoint_dir
        reward_fn = build_reward_funcs(tcfg.reward)

        runtime = ctx.runtime
        ppo_cfg_kwargs = _filter_kwargs(
            PPOConfig,
            {
                "output_dir": out_dir,
                "learning_rate": runtime.learning_rate,
                "per_device_train_batch_size": int(tcfg.per_device_train_batch_size),
                "gradient_accumulation_steps": runtime.gradient_accumulation_steps,
                "gradient_checkpointing": runtime.gradient_checkpointing,
                "lr_scheduler_type": runtime.lr_scheduler_type,
                "warmup_ratio": runtime.warmup_ratio,
                "weight_decay": runtime.weight_decay,
                "max_grad_norm": runtime.max_grad_norm,
                "fp16": runtime.mixed_precision == "fp16",
                "bf16": runtime.mixed_precision == "bf16",
                "num_train_epochs": float(tcfg.get("num_train_epochs", 1)),
                "logging_steps": int(tcfg.get("logging_steps", 5)),
                "save_strategy": str(tcfg.get("save_strategy", "no")),
                "seed": runtime.seed,
                "batch_size": int(tcfg.per_device_train_batch_size),
                "mini_batch_size": int(tcfg.get("ppo", {}).get("mini_batch_size", 1)),
                "kl_coef": float(tcfg.get("ppo", {}).get("kl_coef", 0.05))
                if tcfg.get("ppo")
                else 0.05,
            },
        )
        ppo_config = PPOConfig(**ppo_cfg_kwargs) if ppo_cfg_kwargs else PPOConfig()

        trainer_kwargs = _filter_kwargs(
            PPOTrainer.__init__,
            {
                "config": ppo_config,
                "args": ppo_config,
                "model": model,
                "ref_model": None,
                "tokenizer": tokenizer,
                "processing_class": tokenizer,
                "train_dataset": train_ds,
                "dataset": train_ds,
            },
        )
        # __init__ 的 self 不在签名过滤结果里
        trainer = PPOTrainer(**trainer_kwargs)

        if hasattr(trainer, "train") and "train_dataset" in trainer_kwargs:
            logger.info("PPO 基线：调用 TRL PPOTrainer.train()")
            try:
                train_out = trainer.train(resume_from_checkpoint=ctx.resume_from)
            except TypeError:
                train_out = trainer.train()
            metrics = dict(train_out.metrics) if train_out and getattr(train_out, "metrics", None) else {}
        else:
            logger.info("PPO 基线：调用 TRL PPOTrainer.step() 循环")
            metrics = _classic_step_loop(
                trainer, tokenizer, train_ds, reward_fn, tcfg
            )

        metrics["ppo/baseline"] = 1.0
        self.save(getattr(trainer, "model", model), tokenizer, out_dir)
        return TrainResult(metrics=metrics, checkpoint_dir=out_dir, extra={"impl": "trl"})


def _classic_step_loop(trainer: Any, tokenizer: Any, train_ds: Any, reward_fn: Any, tcfg: Any) -> dict[str, float]:
    """旧版 TRL PPO：generate + reward + trainer.step。"""
    import torch

    if not hasattr(trainer, "step"):
        raise RuntimeError("当前 TRL PPOTrainer 既没有 train() 也没有 step()，无法作为基线运行")

    max_new = int(tcfg.get("max_completion_length", 128))
    reward_acc = 0.0
    n = 0
    last_stats: dict[str, float] = {}
    # 逐条跑，避免基线封装绑死某种 batch collator
    for i in range(len(train_ds)):
        row = train_ds[i]
        prompt = render_chat(tokenizer, list(row["prompt"]), add_generation_prompt=True)
        query = tokenizer(prompt, return_tensors="pt")
        query_ids = query["input_ids"]
        if hasattr(trainer, "generate"):
            response_ids = trainer.generate(query_ids, max_new_tokens=max_new)
        else:
            response_ids = trainer.model.generate(
                query_ids.to(trainer.model.device),
                max_new_tokens=max_new,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(response_ids[0], skip_special_tokens=True)
        score = reward_fn(
            [[{"role": "assistant", "content": text}]],
            ground_truth=[row.get("ground_truth")],
        )[0]
        reward = torch.tensor([float(score)])
        stats = trainer.step([query_ids[0]], [response_ids[0]], [reward])
        if isinstance(stats, dict):
            last_stats = {f"ppo/{k}": float(v) for k, v in stats.items() if isinstance(v, (int, float))}
        reward_acc += float(score)
        n += 1
    last_stats["reward/mean"] = reward_acc / max(n, 1)
    last_stats["train/steps"] = float(n)
    return last_stats
