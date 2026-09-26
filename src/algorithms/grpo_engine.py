"""自研 GRPO 训练步：采样一组回答 → 打分 → 组内优势 → clip 策略更新。"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any, Callable

import torch
from torch.optim import AdamW

from src.algorithms.grpo_loss import (
    advantages_from_rewards,
    completion_mask_from_prompt_lens,
    compute_grpo_loss,
    sequence_logprobs_from_logits,
)
from src.data.chat_templates import render_chat

logger = logging.getLogger("posttrainlab.grpo")


def _device_of(model: Any) -> torch.device:
    return next(model.parameters()).device


def _as_text(completion: str) -> list[dict[str, str]]:
    return [{"role": "assistant", "content": completion}]


class GRPOEngine:
    """不依赖 TRL 的 GRPO 更新器。"""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        reward_fn: Callable[..., list[float]],
        *,
        num_generations: int = 4,
        max_completion_length: int = 256,
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
        # 优化器、调度器、混合精度由 BaseTrainer 注入；单独调用时才自建 AdamW
        self.optimizer = optimizer or AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=float(learning_rate),
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

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

        step = 0
        micro = 0
        last_metrics: dict[str, float] = {}
        reward_sum = 0.0
        reward_n = 0

        for _epoch in range(int(epochs)):
            for start in range(0, len(rows), int(batch_size)):
                batch = rows[start : start + int(batch_size)]
                if not batch:
                    continue
                metrics = self._update_batch(batch, device, grad_accum)
                micro += 1
                reward_sum += metrics.get("reward/mean", 0.0)
                reward_n += 1
                last_metrics = metrics
                if micro % grad_accum == 0:
                    self._optimizer_step()
                    step += 1
                    if logging_steps and step % int(logging_steps) == 0:
                        logger.info(
                            "grpo step=%s loss=%.4f reward=%.4f kl=%.4f",
                            step,
                            metrics.get("loss", 0.0),
                            metrics.get("reward/mean", 0.0),
                            metrics.get("policy/kl", 0.0),
                        )
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
    ) -> dict[str, float]:
        prompts = [row["prompt"] for row in batch]
        ground_truth = [row.get("ground_truth") for row in batch]
        packed = self._sample_group(prompts, device)
        rewards = self._score(packed["texts"], ground_truth)
        reward_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        advantages = advantages_from_rewards(reward_t, self.num_generations)

        # 采样策略 logprob（无梯度）与参考策略 logprob
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

        metrics["reward/mean"] = float(reward_t.mean().item())
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

        metrics["reward/mean"] = float(reward_t.mean().item())
        return metrics

    def _sample_group(self, prompts: list, device: torch.device) -> dict[str, Any]:
        """每个 prompt 采样 G 条 completion，并右填充成张量。"""
        self.tokenizer.padding_side = "left"
        texts = [
            render_chat(self.tokenizer, list(messages), add_generation_prompt=True)
            for messages in prompts
        ]
        enc = self.tokenizer(texts, return_tensors="pt", padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        prompt_width = enc["input_ids"].shape[1]
        prompt_lens = enc["attention_mask"].sum(dim=1)  # 不含左侧 pad

        self.model.eval()
        with torch.no_grad():
            generated = self.model.generate(
                **enc,
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                temperature=max(self.temperature, 1e-5),
                num_return_sequences=self.num_generations,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        self.model.train()

        # generate 会把同一 prompt 重复 G 次；左填充后 completion 从 prompt_width 开始
        g = self.num_generations
        prompt_lens_rep = prompt_lens.repeat_interleave(g)
        completions = []
        decoded = []
        for i in range(generated.shape[0]):
            comp_ids = generated[i, prompt_width:]
            # 去掉尾部 pad
            if self.tokenizer.pad_token_id is not None:
                nonzero = (comp_ids != self.tokenizer.pad_token_id).nonzero(as_tuple=False)
                if nonzero.numel() == 0:
                    comp_ids = comp_ids[:0]
                else:
                    comp_ids = comp_ids[: int(nonzero[-1]) + 1]
            completions.append(comp_ids)
            decoded.append(self.tokenizer.decode(comp_ids, skip_special_tokens=True))

        # 右填充：prompt（去左 pad）+ completion
        sequences = []
        plen_unpadded = []
        for i, comp in enumerate(completions):
            src = i // g
            plen = int(prompt_lens[src].item())
            prompt_ids = enc["input_ids"][src, -plen:]
            sequences.append(torch.cat([prompt_ids, comp], dim=0))
            plen_unpadded.append(plen)

        pad_id = self.tokenizer.pad_token_id
        max_len = max(int(s.shape[0]) for s in sequences)
        input_ids = torch.full((len(sequences), max_len), pad_id, dtype=torch.long, device=device)
        attn = torch.zeros((len(sequences), max_len), dtype=torch.long, device=device)
        for i, seq in enumerate(sequences):
            input_ids[i, : seq.shape[0]] = seq
            attn[i, : seq.shape[0]] = 1

        prompt_lens_t = torch.tensor(plen_unpadded, device=device, dtype=torch.long)
        target_attn = attn[:, 1:]
        mask = completion_mask_from_prompt_lens(input_ids.shape[1] - 1, prompt_lens_t, target_attn)
        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "mask": mask,
            "texts": decoded,
            "prompt_lens_rep": prompt_lens_rep,
        }

    def _score(self, texts: list[str], ground_truth: list[Any]) -> list[float]:
        g = self.num_generations
        gt = []
        for label in ground_truth:
            gt.extend([label] * g)
        completions = [_as_text(t) for t in texts]
        scores = self.reward_fn(completions, ground_truth=gt)
        if len(scores) != len(texts):
            raise RuntimeError(f"奖励条数 {len(scores)} 与生成条数 {len(texts)} 不一致")
        return [float(s) for s in scores]

    def _token_logps(self, packed: dict[str, Any], device: torch.device) -> torch.Tensor:
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
        if hasattr(self.model, "disable_adapter"):
            was_training = self.model.training
            self.model.eval()
            with self.model.disable_adapter(), torch.no_grad():
                ref = self._token_logps(packed, device)
            if was_training:
                self.model.train()
            return ref
        return old_logps
