"""基于 TRL SFTTrainer 的封装：接入自研采样器与梯度裁剪。"""

from __future__ import annotations

import logging
from typing import Any

from transformers import TrainerCallback
from trl import SFTTrainer as TRLSFTTrainer

from src.algorithms.grad_clip import GradClipper, build_grad_clipper
from src.algorithms.sft_samplers import build_sft_sampler, estimate_dataset_lengths

logger = logging.getLogger("posttrainlab.sft")


class GradClipCallback(TrainerCallback):
    """在 optimizer.step 前应用自研裁剪（HF 在 max_grad_norm=0 时不会自己裁）。"""

    def __init__(self, clipper: GradClipper):
        self.clipper = clipper

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if model is None or self.clipper is None:
            return
        self.clipper.clip(model)


class PostTrainSFTTrainer(TRLSFTTrainer):
    """
    TRL SFTTrainer 子类。

    自研点：
    - 自定义 / 长度分组 / 分桶采样（``_get_train_sampler``）
    - 可切换的梯度裁剪策略（``GradClipCallback`` + ``clip_grad_norm_``）
    """

    def __init__(
        self,
        *args: Any,
        sampler_config: dict[str, Any] | None = None,
        grad_clipper: GradClipper | None = None,
        length_cache: list[int] | None = None,
        **kwargs: Any,
    ):
        self.sampler_config = dict(sampler_config or {})
        self.grad_clipper = grad_clipper
        self._length_cache = length_cache
        callbacks = list(kwargs.pop("callbacks", None) or [])
        if grad_clipper is not None:
            callbacks.append(GradClipCallback(grad_clipper))
        if callbacks:
            kwargs["callbacks"] = callbacks
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self, train_dataset: Any = None, *args: Any, **kwargs: Any):
        dataset = train_dataset if train_dataset is not None else getattr(self, "train_dataset", None)
        mode = str(self.sampler_config.get("mode", "random")).lower()
        if dataset is None or mode in ("random", "default", "hf", "none", ""):
            return self._call_super_train_sampler(train_dataset, *args, **kwargs)

        batch_size = int(getattr(self.args, "per_device_train_batch_size", 1) or 1)
        seed = int(self.sampler_config.get("seed", getattr(self.args, "seed", 0) or 0))
        max_length = int(
            self.sampler_config.get(
                "max_length",
                getattr(self.args, "max_length", None)
                or getattr(self.args, "max_seq_length", 2048)
                or 2048,
            )
        )
        tokenizer = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        if self._length_cache is None:
            logger.info("预计算 SFT 样本长度用于采样 mode=%s", mode)
            self._length_cache = estimate_dataset_lengths(
                dataset, max_length=max_length, tokenizer=tokenizer
            )

        sampler = build_sft_sampler(
            dataset,
            mode=mode,
            batch_size=batch_size,
            mega_batch_mult=int(self.sampler_config.get("mega_batch_mult", 8)),
            boundaries=self.sampler_config.get("bucket_boundaries"),
            seed=seed,
            max_length=max_length,
            tokenizer=tokenizer,
            lengths=self._length_cache,
            drop_last=bool(self.sampler_config.get("drop_last", False)),
        )
        logger.info(
            "使用自研采样器 mode=%s batch_size=%s mega_mult=%s n=%s",
            mode,
            batch_size,
            self.sampler_config.get("mega_batch_mult", 8),
            len(dataset),
        )
        return sampler

    def _call_super_train_sampler(self, train_dataset: Any, *args: Any, **kwargs: Any):
        parent = super()._get_train_sampler
        try:
            if train_dataset is not None:
                return parent(train_dataset, *args, **kwargs)
            return parent(*args, **kwargs)
        except TypeError:
            return parent()

    def clip_grad_norm_(self, parameters: Any, *args: Any, **kwargs: Any):
        if self.grad_clipper is not None:
            return self.grad_clipper.clip(parameters)
        parent = getattr(super(), "clip_grad_norm_", None)
        if callable(parent):
            return parent(parameters, *args, **kwargs)
        import torch

        max_norm = float(getattr(self.args, "max_grad_norm", 1.0) or 1.0)
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    def log(self, logs: dict[str, float], *args: Any, **kwargs: Any) -> None:
        if self.grad_clipper is not None:
            logs = dict(logs)
            logs.update(self.grad_clipper.metrics())
        try:
            return super().log(logs, *args, **kwargs)
        except TypeError:
            return super().log(logs)


def build_posttrain_sft_trainer(
    *,
    model: Any,
    args: Any,
    train_dataset: Any,
    eval_dataset: Any = None,
    tokenizer: Any = None,
    sampler_config: dict[str, Any] | None = None,
    grad_clip_raw: Any = None,
    default_max_norm: float = 1.0,
) -> PostTrainSFTTrainer:
    """工厂：组装 PostTrainSFTTrainer（含裁剪器）。"""
    clipper = None
    if grad_clip_raw is not None:
        strategy = str(
            grad_clip_raw.get("strategy", "norm")
            if hasattr(grad_clip_raw, "get")
            else "norm"
        ).lower()
        if strategy not in ("hf", "default"):
            clipper = build_grad_clipper(grad_clip_raw, default_max_norm=default_max_norm)
    return PostTrainSFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        sampler_config=sampler_config,
        grad_clipper=clipper,
    )
