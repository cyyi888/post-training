"""PPO 基线：TRL PPOTrainer + 自研奖励；可选在线 generate→reward→step 循环。"""

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


def _row_prompt_key(row: dict[str, Any]) -> str:
    prompt = row.get("prompt")
    if isinstance(prompt, list) and prompt:
        return str(prompt[-1].get("content", prompt))
    return str(prompt or "")


def _trl_reward_adapter(reward_fn: Any, train_ds: Any):
    """把自研 RewardEngine 适配成 TRL 可调用对象，并绑定 dataset 的 ground_truth。"""
    gt_by_prompt: dict[str, Any] = {}
    meta_by_prompt: dict[str, dict[str, Any]] = {}
    try:
        for i in range(len(train_ds)):
            row = train_ds[i]
            key = _row_prompt_key(row)
            gt_by_prompt[key] = row.get("ground_truth")
            meta_by_prompt[key] = {
                "task_prompt": row.get("task_prompt"),
                "test": row.get("test"),
                "entry_point": row.get("entry_point"),
            }
    except Exception:
        pass

    def _fn(prompts=None, completions=None, **kwargs):
        texts = completions if completions is not None else prompts
        if texts is None:
            return []
        wrapped = []
        for t in texts:
            if isinstance(t, list) and t and isinstance(t[0], dict):
                wrapped.append(t)
            else:
                wrapped.append([{"role": "assistant", "content": str(t)}])

        ground_truth = kwargs.get("ground_truth")
        task_prompts = kwargs.get("prompts")
        tests = kwargs.get("tests")
        entry_points = kwargs.get("entry_points")
        if ground_truth is None and prompts is not None:
            ground_truth = []
            task_prompts = []
            tests = []
            entry_points = []
            for p in prompts:
                key = p if isinstance(p, str) else str(p)
                if isinstance(p, list) and p and isinstance(p[0], dict):
                    key = str(p[-1].get("content", p))
                ground_truth.append(gt_by_prompt.get(key))
                meta = meta_by_prompt.get(key, {})
                task_prompts.append(meta.get("task_prompt") or key)
                tests.append(meta.get("test"))
                entry_points.append(meta.get("entry_point"))

        call_kwargs = dict(kwargs)
        if ground_truth is not None:
            call_kwargs["ground_truth"] = ground_truth
        if task_prompts is not None:
            call_kwargs["prompts"] = task_prompts
        if tests is not None:
            call_kwargs["tests"] = tests
        if entry_points is not None:
            call_kwargs["entry_points"] = entry_points
        return reward_fn(wrapped, **call_kwargs)

    return _fn


@register(ALGORITHM_REGISTRY, "ppo")
class PPOStageTrainer(BaseTrainer):
    """
    基于 TRL 的 PPO 基线封装，奖励与 GRPO 共用 ``RewardEngine``。

    默认 ``ppo.online_loop=true``：自研在线循环（generate → 奖励 → ``PPOTrainer.step``），
    保证 accuracy/steps/code 等依赖 ``ground_truth`` / 测例元数据的奖励可用。
    完整 6 步 Rollout + 组内优势仍以自研 GRPO 为主路径。
    """

    name = "ppo"

    def run_algorithm(self, ctx: TrainContext) -> TrainResult:
        train_ds = ctx.train_dataset
        model = self.load_adapter_resume(ctx.model, ctx.resume_from)
        tokenizer = ctx.tokenizer
        PPOConfig, PPOTrainer = _import_trl_ppo()
        tcfg = ctx.config.training
        out_dir = ctx.checkpoint_dir
        reward_fn = build_reward_funcs(tcfg.reward)
        trl_reward = _trl_reward_adapter(reward_fn, train_ds)

        runtime = ctx.runtime
        ppo_section = tcfg.get("ppo") or {}
        online_loop = bool(ppo_section.get("online_loop", True))
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
                "mini_batch_size": int(ppo_section.get("mini_batch_size", 1)),
                "kl_coef": float(ppo_section.get("kl_coef", 0.05)),
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
                "reward_fns": [trl_reward],
                "reward_funcs": [trl_reward],
                "reward_fn": trl_reward,
                "reward_model": None,
            },
        )
        trainer = PPOTrainer(**trainer_kwargs)

        if online_loop or not (hasattr(trainer, "train") and "train_dataset" in trainer_kwargs):
            logger.info(
                "PPO 基线：在线 RL 循环（generate → RewardEngine → step）online_loop=%s",
                online_loop,
            )
            metrics = _online_rl_loop(trainer, tokenizer, train_ds, reward_fn, tcfg)
            metrics["ppo/online_loop"] = 1.0
        else:
            logger.info("PPO 基线：调用 TRL PPOTrainer.train()")
            try:
                train_out = trainer.train(resume_from_checkpoint=ctx.resume_from)
            except TypeError:
                train_out = trainer.train()
            metrics = (
                dict(train_out.metrics)
                if train_out and getattr(train_out, "metrics", None)
                else {}
            )
            metrics["ppo/online_loop"] = 0.0

        metrics["ppo/baseline"] = 1.0
        if ctx.init_from:
            metrics["ppo/init_from"] = 1.0
        self.save(getattr(trainer, "model", model), tokenizer, out_dir)
        return TrainResult(metrics=metrics, checkpoint_dir=out_dir, extra={"impl": "trl"})


def _online_rl_loop(
    trainer: Any,
    tokenizer: Any,
    train_ds: Any,
    reward_fn: Any,
    tcfg: Any,
) -> dict[str, float]:
    """在线 RL：逐条 generate → 自研奖励（含 gt/测例）→ ``trainer.step``。"""
    import torch

    if not hasattr(trainer, "step"):
        raise RuntimeError("当前 TRL PPOTrainer 没有 step()，无法跑在线循环；请改用 training=grpo")

    max_new = int(tcfg.get("max_completion_length", 128))
    max_steps = tcfg.get("max_steps", None)
    max_steps_i = None if max_steps in (None, "null") else int(max_steps)
    reward_acc = 0.0
    n = 0
    last_stats: dict[str, float] = {}
    model = getattr(trainer, "model", None)
    device = next(model.parameters()).device if model is not None else torch.device("cpu")

    for i in range(len(train_ds)):
        if max_steps_i is not None and n >= max_steps_i:
            break
        row = train_ds[i]
        prompt = render_chat(tokenizer, list(row["prompt"]), add_generation_prompt=True)
        query = tokenizer(prompt, return_tensors="pt")
        query_ids = query["input_ids"].to(device)
        gen_model = model if model is not None else trainer
        if hasattr(trainer, "generate"):
            response_ids = trainer.generate(query_ids, max_new_tokens=max_new)
        else:
            response_ids = gen_model.generate(
                query_ids,
                max_new_tokens=max_new,
                pad_token_id=tokenizer.pad_token_id,
            )
        # 只解码 completion，避免把 prompt 算进奖励文本
        prompt_len = int(query_ids.shape[-1])
        completion_ids = response_ids[0][prompt_len:]
        text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        score = reward_fn(
            [[{"role": "assistant", "content": text}]],
            ground_truth=[row.get("ground_truth")],
            prompts=[row.get("task_prompt") or _row_prompt_key(row)],
            tests=[row.get("test")],
            entry_points=[row.get("entry_point")],
        )[0]
        reward = torch.tensor([float(score)], device=device)
        q = query_ids[0].detach().cpu()
        r = response_ids[0].detach().cpu()
        stats = trainer.step([q], [r], [reward.detach().cpu()])
        if isinstance(stats, dict):
            last_stats = {
                f"ppo/{k}": float(v) for k, v in stats.items() if isinstance(v, (int, float))
            }
        reward_acc += float(score)
        n += 1

    last_stats["reward/mean"] = reward_acc / max(n, 1)
    last_stats["train/steps"] = float(n)
    return last_stats
