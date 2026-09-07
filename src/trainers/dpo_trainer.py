from __future__ import annotations

from typing import Any

from trl import DPOConfig, DPOTrainer

from src.algorithms.dpo_loss import DPOLossConfig
from src.core.base import BaseTrainer, TrainResult
from src.core.registry import ALGORITHM_REGISTRY, register


@register(ALGORITHM_REGISTRY, "dpo")
class DPOStageTrainer(BaseTrainer):
    """
    DPO trainer wrapping TRL.

    Custom loss knobs (dynamic beta / penalty) are recorded in
    ``DPOLossConfig`` for ablations and for the standalone ``DPOLoss`` module.
    Full custom-loss injection into TRL is left as an extension point.
    """

    def __init__(self, cfg: Any, model: Any, tokenizer: Any, dataset: Any, ref_model: Any = None):
        super().__init__(cfg, model, tokenizer, dataset)
        self.ref_model = ref_model
        t = cfg.training
        pen = t.get("penalty", {})
        self.loss_cfg = DPOLossConfig(
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

    def train(self) -> TrainResult:
        tcfg = self.cfg.training
        out_dir = f"{self.cfg.output_dir}/{tcfg.output_subdir}"
        args = DPOConfig(
            output_dir=out_dir,
            beta=tcfg.beta,
            loss_type="ipo" if tcfg.get("loss_type") == "ipo" else "sigmoid",
            learning_rate=tcfg.learning_rate,
            num_train_epochs=tcfg.num_train_epochs,
            per_device_train_batch_size=tcfg.per_device_train_batch_size,
            gradient_accumulation_steps=tcfg.gradient_accumulation_steps,
            gradient_checkpointing=tcfg.gradient_checkpointing,
            logging_steps=tcfg.logging_steps,
            max_length=tcfg.max_length,
            max_prompt_length=tcfg.get("max_prompt_length", 256),
            seed=self.cfg.seed,
            report_to=self.cfg.get("report_to", "none"),
            save_strategy=tcfg.get("save_strategy", "epoch"),
        )
        trainer = DPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            args=args,
            train_dataset=self.dataset,
            processing_class=self.tokenizer,
        )
        train_out = trainer.train()
        metrics = dict(train_out.metrics) if train_out and train_out.metrics else {}
        metrics["dpo/configured_beta"] = self.loss_cfg.beta
        metrics["dpo/dynamic_beta"] = float(self.loss_cfg.dynamic_beta)
        self.model = trainer.model
        return TrainResult(
            metrics=metrics,
            checkpoint_dir=out_dir,
            extra={"loss_cfg": self.loss_cfg},
        )
