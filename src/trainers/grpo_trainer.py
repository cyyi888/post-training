"""GRPO 训练策略：完整自研实现（不调用 TRL GRPOTrainer）。"""

from __future__ import annotations

import logging

from src.algorithms.grpo_engine import GRPOEngine
from src.algorithms.reward_functions import build_reward_funcs
from src.core.base import BaseTrainer, TrainContext, TrainResult
from src.core.registry import ALGORITHM_REGISTRY, register

logger = logging.getLogger("posttrainlab.grpo")


@register(ALGORITHM_REGISTRY, "grpo")
class GRPOStageTrainer(BaseTrainer):
    """自研 GRPO。采样与损失在算法模块，日志/续训/分布式在基类。"""

    name = "grpo"

    def run_algorithm(self, ctx: TrainContext) -> TrainResult:
        train_ds = ctx.train_dataset
        rows = [train_ds[i] for i in range(len(train_ds))]
        model = self.prepare_native_model(ctx.model, ctx.resume_from)
        reward_fn = build_reward_funcs(ctx.config.training.reward)
        tcfg = ctx.config.training

        runtime = ctx.runtime
        batch_size = int(tcfg.get("per_device_train_batch_size", 1))
        epochs = int(tcfg.get("num_train_epochs", 1))
        update_steps = self.estimate_update_steps(
            len(rows),
            batch_size,
            epochs,
            runtime.gradient_accumulation_steps,
        )
        optimizer, scheduler, scaler = self.build_optimizer_and_scheduler(model, update_steps)
        engine = GRPOEngine(
            model,
            ctx.tokenizer,
            reward_fn,
            num_generations=int(tcfg.get("num_generations", 4)),
            max_completion_length=int(tcfg.get("max_completion_length", 256)),
            temperature=float(tcfg.get("temperature", 0.9)),
            clip_range=float(tcfg.get("clip_range", 0.2)),
            beta=float(tcfg.get("beta", 0.04)),
            learning_rate=runtime.learning_rate,
            num_iterations=int(tcfg.get("num_iterations", 1)),
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            mixed_precision=runtime.mixed_precision,
            max_grad_norm=runtime.max_grad_norm,
        )
        max_steps = tcfg.get("max_steps", None)
        metrics = engine.train_loop(
            rows,
            epochs=epochs,
            batch_size=batch_size,
            grad_accum=runtime.gradient_accumulation_steps,
            max_steps=None if max_steps in (None, "null") else int(max_steps),
            logging_steps=int(tcfg.get("logging_steps", 5)),
        )
        out_dir = ctx.checkpoint_dir
        self.save(model, ctx.tokenizer, out_dir)
        logger.info("自研 GRPO 完成 steps=%s", metrics.get("train/steps"))
        return TrainResult(metrics=metrics, checkpoint_dir=out_dir, extra={"impl": "native"})
