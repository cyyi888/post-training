"""DPO / IPO 训练策略。"""

from __future__ import annotations

from trl import DPOConfig
from trl import DPOTrainer as TRLDPOTrainer

from src.algorithms.dpo_loss import DPOLossConfig
from src.core.base import BaseTrainer, TrainContext, TrainResult
from src.core.registry import ALGORITHM_REGISTRY, register


@register(ALGORITHM_REGISTRY, "dpo")
class DPOStageTrainer(BaseTrainer):
    """偏好对齐策略（DPO/IPO）。只实现算法，工程能力在 BaseTrainer。"""

    name = "dpo"

    def run_algorithm(self, ctx: TrainContext) -> TrainResult:
        train_ds = ctx.train_dataset
        t = ctx.config.training
        pen = t.get("penalty", {})
        loss_cfg = DPOLossConfig(
            loss_type=t.get("loss_type", "dpo"),
            beta=t.beta,
            dynamic_beta=t.get("dynamic_beta", False),
            beta_min=t.get("beta_min", 0.05),
            beta_max=t.get("beta_max", 0.5),
            label_smoothing=t.get("label_smoothing", 0.0),
            penalty_enabled=pen.get("enabled", False) if pen else False,
            penalty_type=pen.get("type", "none") if pen else "none",
            penalty_weight=pen.get("weight", 0.0) if pen else 0.0,
        )

        out_dir = ctx.checkpoint_dir
        args = DPOConfig(
            output_dir=out_dir,
            beta=t.beta,
            loss_type="ipo" if t.get("loss_type") == "ipo" else "sigmoid",
            num_train_epochs=t.num_train_epochs,
            per_device_train_batch_size=t.per_device_train_batch_size,
            logging_steps=t.logging_steps,
            max_length=t.max_length,
            max_prompt_length=t.get("max_prompt_length", 256),
            save_strategy=t.get("save_strategy", "epoch"),
            **ctx.hf_args,
        )
        trainer = TRLDPOTrainer(
            model=ctx.model,
            ref_model=None,
            args=args,
            train_dataset=train_ds,
            processing_class=ctx.tokenizer,
        )
        train_out = trainer.train(resume_from_checkpoint=ctx.resume_from)
        metrics = dict(train_out.metrics) if train_out and train_out.metrics else {}
        metrics["dpo/configured_beta"] = loss_cfg.beta
        metrics["dpo/dynamic_beta"] = float(loss_cfg.dynamic_beta)
        self.save(trainer.model, ctx.tokenizer, out_dir)
        return TrainResult(
            metrics=metrics,
            checkpoint_dir=out_dir,
            extra={"loss_cfg": loss_cfg},
        )
