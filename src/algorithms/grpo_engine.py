"""自研 GRPO 训练步：Rollout 管线 → clip 策略更新。"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any, Callable

import torch
from torch.optim import AdamW

from src.algorithms.grpo_loss import compute_grpo_loss, sequence_logprobs_from_logits
from src.algorithms.rollout import (
    GRPORolloutPipeline,
    RolloutConfig,
    iter_dynamic_batches,
    prompt_token_lengths,
    unwrap_generate_model,
)

logger = logging.getLogger("posttrainlab.grpo")


def _device_of(model: Any) -> torch.device:
    return next(model.parameters()).device


class GRPOEngine:
    """不依赖 TRL 的 GRPO 更新器；采样与筛选委托给 ``GRPORolloutPipeline``。"""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        reward_fn: Callable[..., list[float]],
        *,
        num_generations: int = 4,
        max_completion_length: int = 512,
        temperature: float = 0.9,
        clip_range: float = 0.2,
        beta: float = 0.04,
        learning_rate: float = 2e-6,
        num_iterations: int = 1,
        optimizer: Any = None,
        scheduler: Any = None,
        scaler: Any = None,
        mixed_precision: str = "no",
        max_grad_norm: float = 1.0,
        rollout_config: RolloutConfig | None = None,
        log_fn: Callable[..., None] | None = None,
        seed: int = 0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.num_generations = int(num_generations)
        self.max_completion_length = int(max_completion_length)
        self.temperature = float(temperature)
        self.clip_range = float(clip_range)
        self.beta = float(beta)
        self.num_iterations = max(1, int(num_iterations))
        self.mixed_precision = str(mixed_precision)
        self.max_grad_norm = float(max_grad_norm)
        self.scheduler = scheduler
        self.scaler = scaler
        self.seed = int(seed)
        self.log_fn = log_fn
        self.optimizer = optimizer or AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=float(learning_rate),
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        if rollout_config is None:
            rollout_config = RolloutConfig(
                num_generations=self.num_generations,
                max_completion_length=self.max_completion_length,
                temperature=self.temperature,
            )
        else:
            rollout_config.num_generations = self.num_generations
            rollout_config.max_completion_length = self.max_completion_length
            rollout_config.temperature = self.temperature
        self.rollout_config = rollout_config
        self.pipeline = GRPORolloutPipeline(
            reward_fn,
            rollout_config,
            log_fn=log_fn,
        )

    def train_loop(
        self,
        rows: list[dict[str, Any]],
        *,
        epochs: int = 1,
        batch_size: int = 1,
        grad_accum: int = 1,
        max_steps: int | None = None,
        logging_steps: int = 5,
    ) -> dict[str, float]:
        """对投影后的 GRPO 样本（含 prompt / ground_truth）做在线更新。"""
        self.model.train()
        device = _device_of(self.model)
        grad_accum = max(1, int(grad_accum))
        self.optimizer.zero_grad(set_to_none=True)
        rcfg = self.rollout_config

        step = 0
        micro = 0
        last_metrics: dict[str, float] = {}
        reward_sum = 0.0
        reward_n = 0

        for epoch in range(int(epochs)):
            # 1. 固定档分桶 + 动态 batch
            lengths = prompt_token_lengths(rows, self.tokenizer) if rows else []
            batches = list(
                iter_dynamic_batches(
                    rows,
                    lengths,
                    base_batch_size=batch_size,
                    boundaries=rcfg.bucket_boundaries,
                    bucket_batch_sizes=rcfg.bucket_batch_sizes,
                    bucket=rcfg.bucket,
                    seed=self.seed + epoch,
                )
            )
            for batch in batches:
                if not batch:
                    continue
                metrics = self._update_batch(batch, device, grad_accum, step=step)
                if metrics.get("skipped", 0.0) >= 1.0:
                    continue
                micro += 1
                reward_sum += metrics.get("reward/mean", 0.0)
                reward_n += 1
                last_metrics = metrics
                if micro % grad_accum == 0:
                    self._optimizer_step()
                    step += 1
                    if logging_steps and step % int(logging_steps) == 0:
                        logger.info(
                            "grpo step=%s loss=%.4f reward=%.4f kl=%.4f reject=%.3f tok/s=%.1f",
                            step,
                            metrics.get("loss", 0.0),
                            metrics.get("reward/mean", 0.0),
                            metrics.get("policy/kl", 0.0),
                            metrics.get("rollout/reject_rate", 0.0),
                            metrics.get("rollout/perf/tokens_per_sec", 0.0),
                        )
                        if self.log_fn is not None:
                            train_logs = {
                                "train/loss": metrics.get("loss", 0.0),
                                "train/reward_mean": metrics.get("reward/mean", 0.0),
                                "train/kl": metrics.get("policy/kl", 0.0),
                                "train/clipfrac": metrics.get("policy/clipfrac", 0.0),
                            }
                            try:
                                self.log_fn(train_logs, step=step, step_kind="optimizer_step")
                            except TypeError:
                                try:
                                    self.log_fn(train_logs, step=step)
                                except TypeError:
                                    self.log_fn(train_logs)
                    if max_steps is not None and step >= int(max_steps):
                        last_metrics["train/steps"] = float(step)
                        last_metrics["reward/mean_all"] = reward_sum / max(reward_n, 1)
                        return last_metrics

        if micro % grad_accum != 0:
            self._optimizer_step()
            step += 1

        last_metrics["train/steps"] = float(step)
        last_metrics["reward/mean_all"] = reward_sum / max(reward_n, 1)
        return last_metrics

    def _update_batch(
        self,
        batch: list[dict[str, Any]],
        device: torch.device,
        grad_accum: int,
        *,
        step: int | None = None,
    ) -> dict[str, float]:
        assembled = self.pipeline.run(batch, self.model, self.tokenizer, step=step)
        if assembled is None or assembled.empty:
            return {"skipped": 1.0, "reward/mean": 0.0}

        packed = {
            "input_ids": assembled.input_ids,
            "attention_mask": assembled.attention_mask,
            "mask": assembled.mask,
        }
        advantages = assembled.advantages
        reward_t = assembled.rewards

        with torch.no_grad():
            old_logps = self._token_logps(packed, device)
            ref_logps = self._ref_token_logps(packed, device, old_logps)

        metrics: dict[str, float] = {}
        self.model.train()
        for _ in range(self.num_iterations):
            with self._amp_context():
                policy_logps = self._token_logps(packed, device)
                loss, metrics = compute_grpo_loss(
                    policy_logps,
                    old_logps,
                    advantages,
                    packed["mask"],
                    ref_logps=ref_logps,
                    clip_range=self.clip_range,
                    beta=self.beta,
                )
            scaled = loss / grad_accum
            if self.scaler is not None:
                self.scaler.scale(scaled).backward()
            else:
                scaled.backward()

        metrics["reward/mean"] = float(reward_t.mean().item()) if reward_t.numel() else 0.0
        for k, v in assembled.metrics.items():
            metrics[f"rollout/{k}"] = float(v)
        metrics["skipped"] = 0.0
        return metrics

    def _optimizer_step(self) -> None:
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

    def _amp_context(self):
        if self.mixed_precision in ("fp16", "bf16") and torch.cuda.is_available():
            dtype = torch.float16 if self.mixed_precision == "fp16" else torch.bfloat16
            try:
                return torch.autocast(device_type="cuda", dtype=dtype)
            except TypeError:
                return torch.cuda.amp.autocast(dtype=dtype)
        return nullcontext()

    def _token_logps(self, packed: dict[str, Any], device: torch.device) -> torch.Tensor:
        del device
        out = self.model(
            input_ids=packed["input_ids"],
            attention_mask=packed["attention_mask"],
        )
        return sequence_logprobs_from_logits(out.logits, packed["input_ids"])

    def _ref_token_logps(
        self,
        packed: dict[str, Any],
        device: torch.device,
        old_logps: torch.Tensor,
    ) -> torch.Tensor:
        """LoRA 时用 disable_adapter 近似参考模型；否则退回采样策略（仅 clip，无额外 ref KL）。"""
        if self.beta <= 0:
            return old_logps
        maybe = unwrap_generate_model(self.model)
        if hasattr(maybe, "disable_adapter"):
            was_training = self.model.training
            self.model.eval()
            with maybe.disable_adapter(), torch.no_grad():
                ref = self._token_logps(packed, device)
            if was_training:
                self.model.train()
            return ref
        return old_logps
