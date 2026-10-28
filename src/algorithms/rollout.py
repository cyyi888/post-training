"""GRPO 完整 6 步 Rollout 管线。

1. 预处理与长度分桶（固定档 + 动态 batch）
2. 批量 generate（无梯度，top_p/top_k，清 KV）
3. 后处理 / 格式校验 / 槽位补采 / 整组半量重采
4. 分层奖励（正确性 + 格式 + 长度）
5. 组内归一化与样本组装
6. 指标统计（可同步 WandB）
"""

from __future__ import annotations

import logging
import math
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import torch

from src.algorithms.grpo_loss import completion_mask_from_prompt_lens
from src.algorithms.grpo_utils import group_relative_advantages
from src.algorithms.reward_functions import RewardEngine, has_answer_format
from src.data.chat_templates import render_chat

logger = logging.getLogger("posttrainlab.rollout")

LogFn = Callable[..., None]


@dataclass
class LengthRewardConfig:
    min_tokens: int = 16
    max_tokens: int = 512
    target_min: int = 32
    target_max: int = 384

    def as_dict(self) -> dict[str, int]:
        return {
            "min_tokens": self.min_tokens,
            "max_tokens": self.max_tokens,
            "target_min": self.target_min,
            "target_max": self.target_max,
        }


@dataclass
class RolloutConfig:
    num_generations: int = 4
    max_completion_length: int = 512
    temperature: float = 0.9
    top_p: float = 1.0
    top_k: int = 0
    bucket: bool = True
    bucket_boundaries: list[int] = field(default_factory=lambda: [128, 256, 512])
    # 各长度档 batch size；为空则按 base 自动：短大长小
    bucket_batch_sizes: list[int] | None = None
    max_resample: int = 2
    max_group_resample: int = 2
    min_valid_per_group: int = 2
    require_answer_format: bool = True
    filter_extreme_length: bool = True
    clear_cache: bool = True
    # 训练默认 False 保多样性；消融对比时 True + generation_seed
    fix_generation_seed: bool = False
    generation_seed: int = 0
    length_reward: LengthRewardConfig = field(default_factory=LengthRewardConfig)

    def __post_init__(self) -> None:
        pass

    @classmethod
    def from_mapping(cls, cfg: Any | None = None, **overrides: Any) -> "RolloutConfig":
        """从 Hydra / dict 构造；``overrides`` 覆盖同名字段。"""
        data: dict[str, Any] = {}
        if cfg is not None:
            if hasattr(cfg, "items"):
                data.update(dict(cfg.items()))
            elif isinstance(cfg, dict):
                data.update(cfg)
        data.update({k: v for k, v in overrides.items() if v is not None})

        lr_raw = data.pop("length_reward", None) or {}
        if hasattr(lr_raw, "items"):
            lr_raw = dict(lr_raw.items())
        length_reward = LengthRewardConfig(
            min_tokens=int(lr_raw.get("min_tokens", 16)),
            max_tokens=int(lr_raw.get("max_tokens", 512)),
            target_min=int(lr_raw.get("target_min", 32)),
            target_max=int(lr_raw.get("target_max", 384)),
        )

        bounds = data.get("bucket_boundaries", [128, 256, 512])
        if hasattr(bounds, "__iter__") and not isinstance(bounds, (str, bytes)):
            bounds = [int(b) for b in bounds]
        else:
            bounds = [128, 256, 512]

        bbs = data.get("bucket_batch_sizes", None)
        if bbs is not None and hasattr(bbs, "__iter__") and not isinstance(bbs, (str, bytes)):
            bbs = [int(x) for x in bbs]
        else:
            bbs = None

        return cls(
            num_generations=int(data.get("num_generations", 4)),
            max_completion_length=int(data.get("max_completion_length", 512)),
            temperature=float(data.get("temperature", 0.9)),
            top_p=float(data.get("top_p", 1.0)),
            top_k=int(data.get("top_k", 0)),
            bucket=bool(data.get("bucket", True)),
            bucket_boundaries=list(bounds),
            bucket_batch_sizes=bbs,
            max_resample=int(data.get("max_resample", 2)),
            max_group_resample=int(data.get("max_group_resample", 2)),
            min_valid_per_group=int(data.get("min_valid_per_group", 2)),
            require_answer_format=bool(data.get("require_answer_format", True)),
            filter_extreme_length=bool(data.get("filter_extreme_length", True)),
            clear_cache=bool(data.get("clear_cache", True)),
            fix_generation_seed=bool(data.get("fix_generation_seed", False)),
            generation_seed=int(data.get("generation_seed", 0)),
            length_reward=length_reward,
        )


@dataclass
class CompletionRecord:
    prompt_index: int
    text: str
    token_ids: torch.Tensor
    valid: bool = True
    reason: str = ""
    reward: float = 0.0
    reward_parts: dict[str, float] = field(default_factory=dict)
    resampled: int = 0


@dataclass
class AssembledBatch:
    """供 GRPOEngine 直接做 clip 更新的标准训练样本。"""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    mask: torch.Tensor
    advantages: torch.Tensor
    rewards: torch.Tensor
    texts: list[str]
    prompt_lens: list[int]
    group_sizes: list[int]
    metrics: dict[str, float] = field(default_factory=dict)
    empty: bool = False
    # 拆开字段，便于文档对齐
    prompt_ids: list[torch.Tensor] = field(default_factory=list)
    answer_ids: list[torch.Tensor] = field(default_factory=list)


def unwrap_generate_model(model: Any) -> Any:
    """DDP / DataParallel 包裹时取底层模块做 generate。"""
    try:
        from torch.nn.parallel import DataParallel, DistributedDataParallel

        if isinstance(model, (DistributedDataParallel, DataParallel)):
            return model.module
    except Exception:
        pass
    if type(model).__name__ in ("DistributedDataParallel", "DataParallel") and hasattr(
        model, "module"
    ):
        return model.module
    return model


def _device_of(model: Any) -> torch.device:
    return next(model.parameters()).device


def _user_text(prompt_messages: Any) -> str:
    """从 chat messages 取出用户题面文本（代码题 task_prompt 兜底）。"""
    if isinstance(prompt_messages, str):
        return prompt_messages
    if not prompt_messages:
        return ""
    if isinstance(prompt_messages, list):
        for msg in reversed(prompt_messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return str(msg.get("content", ""))
        last = prompt_messages[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
        return str(last)
    return str(prompt_messages)


def prompt_token_lengths(
    rows: list[dict[str, Any]],
    tokenizer: Any,
) -> list[int]:
    """对 GRPO 投影样本计算 prompt token 长度。"""
    lengths: list[int] = []
    for row in rows:
        text = render_chat(tokenizer, list(row["prompt"]), add_generation_prompt=True)
        ids = tokenizer.encode(text, add_special_tokens=False)
        lengths.append(max(1, len(ids)))
    return lengths


def resolve_bucket_batch_sizes(
    base_batch_size: int,
    n_buckets: int,
    explicit: list[int] | None = None,
) -> list[int]:
    """短桶大 batch、长桶小 batch。"""
    base = max(1, int(base_batch_size))
    if explicit is not None:
        sizes = [max(1, int(x)) for x in explicit]
        if len(sizes) < n_buckets:
            sizes = sizes + [sizes[-1]] * (n_buckets - len(sizes))
        return sizes[:n_buckets]
    # 例如 4 档：base*2, base, max(1,base//2), 1
    sizes: list[int] = []
    for i in range(n_buckets):
        if i == 0:
            sizes.append(max(1, base * 2))
        elif i == 1:
            sizes.append(base)
        elif i == n_buckets - 1:
            sizes.append(1)
        else:
            sizes.append(max(1, base // 2))
    return sizes


def assign_bucket_id(length: int, boundaries: list[int]) -> int:
    for i, b in enumerate(boundaries):
        if length <= b:
            return i
    return len(boundaries)


def iter_dynamic_batches(
    rows: list[dict[str, Any]],
    lengths: list[int],
    *,
    base_batch_size: int = 1,
    boundaries: list[int] | None = None,
    bucket_batch_sizes: list[int] | None = None,
    bucket: bool = True,
    seed: int = 0,
) -> Iterator[list[dict[str, Any]]]:
    """
    固定长度档分桶 + 动态 batch。

    档位默认 0~128 / 129~256 / 257~512 / >512；短档用更大 batch。
    """
    if not rows:
        return
    if not bucket:
        bs = max(1, int(base_batch_size))
        for start in range(0, len(rows), bs):
            yield rows[start : start + bs]
        return

    bounds = sorted(int(b) for b in (boundaries or [128, 256, 512]))
    n_buckets = len(bounds) + 1
    sizes = resolve_bucket_batch_sizes(base_batch_size, n_buckets, bucket_batch_sizes)

    buckets: list[list[int]] = [[] for _ in range(n_buckets)]
    for i, L in enumerate(lengths):
        buckets[assign_bucket_id(int(L), bounds)].append(i)

    rng = random.Random(int(seed))
    batch_lists: list[list[int]] = []
    for bid, idxs in enumerate(buckets):
        if not idxs:
            continue
        order = list(idxs)
        rng.shuffle(order)
        bs = sizes[bid]
        for start in range(0, len(order), bs):
            chunk = order[start : start + bs]
            batch_lists.append(chunk)
    rng.shuffle(batch_lists)
    for chunk in batch_lists:
        yield [rows[i] for i in chunk]


def _strip_completion_ids(comp_ids: torch.Tensor, pad_token_id: int | None) -> torch.Tensor:
    if pad_token_id is None:
        return comp_ids
    nonzero = (comp_ids != pad_token_id).nonzero(as_tuple=False)
    if nonzero.numel() == 0:
        return comp_ids[:0]
    return comp_ids[: int(nonzero[-1]) + 1]


def _clean_text(text: str) -> str:
    # 压缩多余空白，保留换行结构以便 #### 解析
    text = text.replace("\x00", "").replace("\r", "")
    lines = [" ".join(line.split()) for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def validate_completion(
    text: str,
    token_ids: torch.Tensor,
    *,
    require_answer_format: bool = True,
    max_completion_length: int = 512,
    filter_extreme_length: bool = True,
    min_tokens: int = 16,
    max_tokens: int = 512,
) -> tuple[bool, str]:
    """格式校验：空串 / 无答案格式 / 长度极端异常。"""
    n = int(token_ids.numel())
    if n == 0:
        return False, "empty_tokens"
    cleaned = _clean_text(text)
    if not cleaned:
        return False, "empty_text"
    if filter_extreme_length:
        if n < int(min_tokens):
            return False, "too_short"
        if n > int(max_tokens) or n > int(max_completion_length):
            return False, "too_long"
    if require_answer_format and not has_answer_format(cleaned):
        return False, "missing_answer_format"
    return True, ""


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return float(ordered[lo] * (1 - frac) + ordered[hi] * frac)


def _cuda_memory_gb(device: torch.device) -> float:
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0.0
    try:
        return float(torch.cuda.max_memory_allocated(device) / (1024**3))
    except Exception:
        return 0.0


def clear_generation_cache(model: Any, device: torch.device) -> None:
    """生成后清理 KV / CUDA 缓存，降低显存泄漏风险。"""
    gen_model = unwrap_generate_model(model)
    for attr in ("past_key_values", "_past_key_values"):
        if hasattr(gen_model, attr):
            try:
                setattr(gen_model, attr, None)
            except Exception:
                pass
    if hasattr(torch, "compiler") and hasattr(torch.compiler, "reset"):
        pass
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


class GRPORolloutPipeline:
    """固定 6 步的 GRPO Rollout 批处理管线。"""

    def __init__(
        self,
        reward_fn: Any,
        config: RolloutConfig,
        *,
        log_fn: LogFn | None = None,
    ):
        self.reward_fn = reward_fn
        self.config = config
        self.log_fn = log_fn
        self._gen_call_counter = 0

    def run(
        self,
        rows: list[dict[str, Any]],
        model: Any,
        tokenizer: Any,
        *,
        step: int | None = None,
    ) -> AssembledBatch | None:
        """对一批 prompt 行执行完整 Rollout；无可训练样本时返回 empty batch。"""
        if not rows:
            return None
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        pad_token_id = int(tokenizer.pad_token_id)

        device = _device_of(model)
        cfg = self.config
        g = cfg.num_generations
        t0 = time.perf_counter()
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        prompts = [row["prompt"] for row in rows]
        ground_truth = [row.get("ground_truth") for row in rows]
        # 代码奖励所需元数据（与 prompt 行对齐；数学题可为 None）
        task_prompts = [row.get("task_prompt") or _user_text(row.get("prompt")) for row in rows]
        tests = [row.get("test") for row in rows]
        entry_points = [row.get("entry_point") for row in rows]

        # 2. 批量生成
        gen_out = self._generate(
            prompts, model, tokenizer, device, num_return=g, step=step
        )
        gen_tokens = int(gen_out.get("new_tokens", 0))

        # 3. 后处理 + 拒绝采样 / 整组重采
        records, reject_stats = self._postprocess_with_rejection(
            gen_out, model, tokenizer, device, prompts, step=step
        )
        gen_tokens += int(reject_stats.get("extra_new_tokens", 0.0))

        # 4. 分层奖励
        self._score_records(
            records,
            ground_truth,
            task_prompts=task_prompts,
            tests=tests,
            entry_points=entry_points,
        )

        # 5. 组内归一化与组装
        assembled = self._assemble(
            records,
            gen_out["prompt_ids_unpadded"],
            device,
            pad_token_id=pad_token_id,
        )

        elapsed = max(time.perf_counter() - t0, 1e-9)
        # 6. 指标
        metrics = self._compute_metrics(
            records,
            reject_stats,
            assembled,
            elapsed_sec=elapsed,
            gen_tokens=gen_tokens,
            device=device,
        )
        assembled.metrics = metrics
        if self.log_fn is not None:
            prefixed = {f"rollout/{k}": v for k, v in metrics.items()}
            try:
                self.log_fn(prefixed, step=step)
            except TypeError:
                self.log_fn(prefixed)
        return assembled

    def _maybe_seed(self, step: int | None) -> None:
        cfg = self.config
        if not cfg.fix_generation_seed:
            return
        base = int(cfg.generation_seed) + int(step or 0) * 10007 + self._gen_call_counter
        self._gen_call_counter += 1
        torch.manual_seed(base)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(base)

    def _generate_kwargs(self, tokenizer: Any, num_return: int) -> dict[str, Any]:
        cfg = self.config
        kwargs: dict[str, Any] = {
            "max_new_tokens": cfg.max_completion_length,
            "do_sample": True,
            "temperature": max(cfg.temperature, 1e-5),
            "num_return_sequences": num_return,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if cfg.top_p is not None and float(cfg.top_p) < 1.0:
            kwargs["top_p"] = float(cfg.top_p)
        if cfg.top_k is not None and int(cfg.top_k) > 0:
            kwargs["top_k"] = int(cfg.top_k)
        return kwargs

    def _generate(
        self,
        prompts: list,
        model: Any,
        tokenizer: Any,
        device: torch.device,
        *,
        num_return: int,
        step: int | None = None,
    ) -> dict[str, Any]:
        cfg = self.config
        tokenizer.padding_side = "left"
        texts = [
            render_chat(tokenizer, list(messages), add_generation_prompt=True)
            for messages in prompts
        ]
        enc = tokenizer(texts, return_tensors="pt", padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        prompt_width = enc["input_ids"].shape[1]
        prompt_lens = enc["attention_mask"].sum(dim=1)

        gen_model = unwrap_generate_model(model)
        was_training = model.training
        gen_model.eval()
        self._maybe_seed(step)
        with torch.no_grad():
            generated = gen_model.generate(**enc, **self._generate_kwargs(tokenizer, num_return))
        if was_training:
            model.train()
        if cfg.clear_cache:
            clear_generation_cache(model, device)

        prompt_ids_unpadded: list[torch.Tensor] = []
        for i in range(len(prompts)):
            plen = int(prompt_lens[i].item())
            prompt_ids_unpadded.append(enc["input_ids"][i, -plen:].detach())

        new_tokens = int(generated.shape[0] * max(generated.shape[1] - prompt_width, 0))
        return {
            "generated": generated,
            "prompt_width": prompt_width,
            "prompt_lens": prompt_lens,
            "prompt_ids_unpadded": prompt_ids_unpadded,
            "new_tokens": new_tokens,
        }

    def _decode_slot(
        self,
        generated_row: torch.Tensor,
        prompt_width: int,
        tokenizer: Any,
    ) -> tuple[torch.Tensor, str]:
        comp_ids = generated_row[prompt_width:]
        comp_ids = _strip_completion_ids(comp_ids, tokenizer.pad_token_id)
        if (
            tokenizer.eos_token_id is not None
            and comp_ids.numel() > 0
            and int(comp_ids[-1].item()) == int(tokenizer.eos_token_id)
        ):
            comp_ids = comp_ids[:-1]
        text = _clean_text(tokenizer.decode(comp_ids, skip_special_tokens=True))
        return comp_ids.detach().cpu(), text

    def _validate(self, text: str, comp_ids: torch.Tensor) -> tuple[bool, str]:
        cfg = self.config
        lr = cfg.length_reward
        return validate_completion(
            text,
            comp_ids,
            require_answer_format=cfg.require_answer_format,
            max_completion_length=cfg.max_completion_length,
            filter_extreme_length=cfg.filter_extreme_length,
            min_tokens=lr.min_tokens,
            max_tokens=lr.max_tokens,
        )

    def _postprocess_with_rejection(
        self,
        gen_out: dict[str, Any],
        model: Any,
        tokenizer: Any,
        device: torch.device,
        prompts: list,
        *,
        step: int | None = None,
    ) -> tuple[list[CompletionRecord], dict[str, float]]:
        cfg = self.config
        g = cfg.num_generations
        b = len(prompts)
        generated = gen_out["generated"]
        prompt_width = gen_out["prompt_width"]
        half = max(1, math.ceil(g / 2))

        records: list[CompletionRecord] = []
        rejected = 0
        resample_attempts = 0
        group_resample_attempts = 0
        extra_new_tokens = 0
        total_slots = b * g

        for src in range(b):
            group_recs: list[CompletionRecord] = []
            for k in range(g):
                idx = src * g + k
                comp_ids, text = self._decode_slot(generated[idx], prompt_width, tokenizer)
                ok, reason = self._validate(text, comp_ids)
                group_recs.append(
                    CompletionRecord(
                        prompt_index=src,
                        text=text,
                        token_ids=comp_ids,
                        valid=ok,
                        reason="" if ok else reason,
                        resampled=0,
                    )
                )

            # 整组半量重采：有效 < G/2 时整组重新 generate
            group_tries = 0
            while (
                sum(1 for r in group_recs if r.valid) < half
                and group_tries < cfg.max_group_resample
            ):
                group_tries += 1
                group_resample_attempts += 1
                new_out = self._generate(
                    [prompts[src]],
                    model,
                    tokenizer,
                    device,
                    num_return=g,
                    step=step,
                )
                extra_new_tokens += int(new_out.get("new_tokens", 0))
                pw = new_out["prompt_width"]
                group_recs = []
                for k in range(g):
                    comp_ids, text = self._decode_slot(new_out["generated"][k], pw, tokenizer)
                    ok, reason = self._validate(text, comp_ids)
                    group_recs.append(
                        CompletionRecord(
                            prompt_index=src,
                            text=text,
                            token_ids=comp_ids,
                            valid=ok,
                            reason="" if ok else reason,
                            resampled=0,
                        )
                    )

            # 槽位级拒绝采样：补齐仍无效的槽位
            for rec in group_recs:
                resamples = 0
                while not rec.valid and resamples < cfg.max_resample:
                    resamples += 1
                    resample_attempts += 1
                    comp_ids, text = self._resample_one(
                        prompts[src], model, tokenizer, device, step=step
                    )
                    extra_new_tokens += int(comp_ids.numel())
                    ok, reason = self._validate(text, comp_ids)
                    rec.text = text
                    rec.token_ids = comp_ids
                    rec.valid = ok
                    rec.reason = "" if ok else reason
                    rec.resampled = resamples
                if not rec.valid:
                    rejected += 1
                records.append(rec)

        stats = {
            "reject_count": float(rejected),
            "resample_attempts": float(resample_attempts),
            "group_resample_attempts": float(group_resample_attempts),
            "total_slots": float(total_slots),
            "extra_new_tokens": float(extra_new_tokens),
            "half_threshold": float(half),
        }
        return records, stats

    def _resample_one(
        self,
        prompt_messages: list,
        model: Any,
        tokenizer: Any,
        device: torch.device,
        *,
        step: int | None = None,
    ) -> tuple[torch.Tensor, str]:
        cfg = self.config
        tokenizer.padding_side = "left"
        text = render_chat(tokenizer, list(prompt_messages), add_generation_prompt=True)
        enc = tokenizer([text], return_tensors="pt", padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        prompt_width = enc["input_ids"].shape[1]
        gen_model = unwrap_generate_model(model)
        was_training = model.training
        gen_model.eval()
        self._maybe_seed(step)
        with torch.no_grad():
            generated = gen_model.generate(**enc, **self._generate_kwargs(tokenizer, 1))
        if was_training:
            model.train()
        if cfg.clear_cache:
            clear_generation_cache(model, device)
        return self._decode_slot(generated[0], prompt_width, tokenizer)

    def _score_records(
        self,
        records: list[CompletionRecord],
        ground_truth: list[Any],
        *,
        task_prompts: list[Any] | None = None,
        tests: list[Any] | None = None,
        entry_points: list[Any] | None = None,
    ) -> None:
        valid = [r for r in records if r.valid]
        if not valid:
            return
        completions = [[{"role": "assistant", "content": r.text}] for r in valid]
        gt = [ground_truth[r.prompt_index] for r in valid]
        lengths = [int(r.token_ids.numel()) for r in valid]
        lr = self.config.length_reward.as_dict()

        def _gather(src: list[Any] | None) -> list[Any] | None:
            if src is None:
                return None
            return [src[r.prompt_index] for r in valid]

        score_kwargs: dict[str, Any] = {
            "ground_truth": gt,
            "completion_lengths": lengths,
            "length_reward": lr,
            "prompts": _gather(task_prompts),
            "tests": _gather(tests),
            "entry_points": _gather(entry_points),
        }

        if isinstance(self.reward_fn, RewardEngine):
            detailed = self.reward_fn.score_detailed(completions, **score_kwargs)
            totals = detailed["total"]
            parts = detailed["parts"]
            for i, rec in enumerate(valid):
                rec.reward = float(totals[i])
                rec.reward_parts = {name: float(vals[i]) for name, vals in parts.items()}
        else:
            scores = self.reward_fn(completions, **score_kwargs)
            for i, rec in enumerate(valid):
                rec.reward = float(scores[i])

    def _assemble(
        self,
        records: list[CompletionRecord],
        prompt_ids_unpadded: list[torch.Tensor],
        device: torch.device,
        *,
        pad_token_id: int,
    ) -> AssembledBatch:
        cfg = self.config
        by_prompt: dict[int, list[CompletionRecord]] = {}
        for rec in records:
            by_prompt.setdefault(rec.prompt_index, []).append(rec)

        kept_groups: list[list[CompletionRecord]] = []
        for src in sorted(by_prompt.keys()):
            group = [r for r in by_prompt[src] if r.valid]
            if len(group) < cfg.min_valid_per_group:
                continue
            kept_groups.append(group)

        if not kept_groups:
            logger.warning("Rollout 无可训练组（全部被过滤）")
            return AssembledBatch(
                input_ids=torch.zeros(0, 1, dtype=torch.long, device=device),
                attention_mask=torch.zeros(0, 1, dtype=torch.long, device=device),
                mask=torch.zeros(0, 0, dtype=torch.float32, device=device),
                advantages=torch.zeros(0, dtype=torch.float32, device=device),
                rewards=torch.zeros(0, dtype=torch.float32, device=device),
                texts=[],
                prompt_lens=[],
                group_sizes=[],
                empty=True,
            )

        all_rewards: list[float] = []
        all_advs: list[float] = []
        sequences: list[torch.Tensor] = []
        plen_unpadded: list[int] = []
        texts: list[str] = []
        group_sizes: list[int] = []
        prompt_ids_list: list[torch.Tensor] = []
        answer_ids_list: list[torch.Tensor] = []

        for group in kept_groups:
            src = group[0].prompt_index
            rewards_t = torch.tensor([r.reward for r in group], dtype=torch.float32)
            g_eff = len(group)
            adv = group_relative_advantages(rewards_t, num_generations=g_eff)
            all_rewards.extend(float(x) for x in rewards_t.tolist())
            all_advs.extend(float(x) for x in adv.tolist())
            group_sizes.append(g_eff)
            prompt_ids = prompt_ids_unpadded[src].to(device)
            plen = int(prompt_ids.shape[0])
            for rec in group:
                comp = rec.token_ids.to(device)
                sequences.append(torch.cat([prompt_ids, comp], dim=0))
                plen_unpadded.append(plen)
                texts.append(rec.text)
                prompt_ids_list.append(prompt_ids.detach().cpu())
                answer_ids_list.append(comp.detach().cpu())

        max_len = max(int(s.shape[0]) for s in sequences)
        n = len(sequences)
        input_ids = torch.full((n, max_len), pad_token_id, dtype=torch.long, device=device)
        attn = torch.zeros((n, max_len), dtype=torch.long, device=device)
        for i, seq in enumerate(sequences):
            input_ids[i, : seq.shape[0]] = seq
            attn[i, : seq.shape[0]] = 1

        prompt_lens_t = torch.tensor(plen_unpadded, device=device, dtype=torch.long)
        target_attn = attn[:, 1:]
        mask = completion_mask_from_prompt_lens(input_ids.shape[1] - 1, prompt_lens_t, target_attn)
        return AssembledBatch(
            input_ids=input_ids,
            attention_mask=attn,
            mask=mask,
            advantages=torch.tensor(all_advs, dtype=torch.float32, device=device),
            rewards=torch.tensor(all_rewards, dtype=torch.float32, device=device),
            texts=texts,
            prompt_lens=plen_unpadded,
            group_sizes=group_sizes,
            empty=False,
            prompt_ids=prompt_ids_list,
            answer_ids=answer_ids_list,
        )

    def _compute_metrics(
        self,
        records: list[CompletionRecord],
        reject_stats: dict[str, float],
        assembled: AssembledBatch,
        *,
        elapsed_sec: float,
        gen_tokens: int,
        device: torch.device,
    ) -> dict[str, float]:
        valid = [r for r in records if r.valid]
        lengths = [float(r.token_ids.numel()) for r in valid]
        rewards = [r.reward for r in valid]
        total_slots = max(reject_stats.get("total_slots", 0.0), 1.0)
        reject_count = reject_stats.get("reject_count", 0.0)

        metrics: dict[str, float] = {
            "reward/mean": float(statistics.fmean(rewards)) if rewards else 0.0,
            "reward/std": float(statistics.pstdev(rewards)) if len(rewards) > 1 else 0.0,
            "reward/max": float(max(rewards)) if rewards else 0.0,
            "reward/min": float(min(rewards)) if rewards else 0.0,
            "length/mean": float(statistics.fmean(lengths)) if lengths else 0.0,
            "length/p50": _percentile(lengths, 0.5),
            "length/p90": _percentile(lengths, 0.9),
            "reject_rate": float(reject_count / total_slots),
            "valid_frac": float(len(valid) / total_slots),
            "resample_attempts": float(reject_stats.get("resample_attempts", 0.0)),
            "group_resample_attempts": float(reject_stats.get("group_resample_attempts", 0.0)),
            "num_train_samples": float(0 if assembled.empty else assembled.rewards.numel()),
            "num_groups": float(len(assembled.group_sizes)),
            "perf/batch_sec": float(elapsed_sec),
            "perf/tokens_per_sec": float(gen_tokens / max(elapsed_sec, 1e-9)),
            "perf/gen_tokens": float(gen_tokens),
            "perf/peak_mem_gb": _cuda_memory_gb(device),
        }

        part_names: set[str] = set()
        for r in valid:
            part_names.update(r.reward_parts.keys())
        for name in sorted(part_names):
            vals = [r.reward_parts.get(name, 0.0) for r in valid]
            metrics[f"reward/{name}_mean"] = float(statistics.fmean(vals)) if vals else 0.0

        acc_vals = [
            r.reward_parts.get("accuracy", r.reward_parts.get("reward_correct", 0.0))
            for r in valid
        ]
        fmt_vals = [
            r.reward_parts.get("format", r.reward_parts.get("reward_format", 0.0))
            for r in valid
        ]
        if acc_vals:
            metrics["pass_rate"] = float(
                sum(1 for v in acc_vals if v >= 1.0 - 1e-8) / len(acc_vals)
            )
        else:
            metrics["pass_rate"] = 0.0
        if fmt_vals:
            metrics["format_rate"] = float(statistics.fmean(fmt_vals))
        else:
            metrics["format_rate"] = metrics["valid_frac"]

        return metrics
