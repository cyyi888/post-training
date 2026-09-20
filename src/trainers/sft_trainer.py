from __future__ import annotations

from typing import Any

from trl import SFTConfig, SFTTrainer

from src.core.base import BaseTrainer, TrainResult
from src.core.registry import ALGORITHM_REGISTRY, register
from src.trainers.common import trainer_logging_kwargs


@register(ALGORITHM_REGISTRY, "sft")
class SFTStageTrainer(BaseTrainer):
    def __init__(self, cfg: Any, model: Any, tokenizer: Any, dataset: Any, eval_dataset: Any = None):
        super().__init__(cfg, model, tokenizer, dataset)
        self.eval_dataset = eval_dataset

    def train(self) -> TrainResult:
        tcfg = self.cfg.training
        out_dir = f"{self.cfg.output_dir}/{tcfg.output_subdir}"
        args = SFTConfig(
            output_dir=out_dir,
            learning_rate=tcfg.learning_rate,
            num_train_epochs=tcfg.num_train_epochs,
            per_device_train_batch_size=tcfg.per_device_train_batch_size,
            per_device_eval_batch_size=tcfg.get("per_device_eval_batch_size", 1),
            gradient_accumulation_steps=tcfg.gradient_accumulation_steps,
            gradient_checkpointing=tcfg.gradient_checkpointing,
            logging_steps=tcfg.logging_steps,
            eval_strategy=tcfg.get("eval_strategy", "no"),
            save_strategy=tcfg.get("save_strategy", "epoch"),
            max_length=tcfg.max_length,
            assistant_only_loss=tcfg.get("assistant_only_loss", True),
            seed=self.cfg.seed,
            **trainer_logging_kwargs(self.cfg),
        )
        trainer = SFTTrainer(
            model=self.model,
            args=args,
            train_dataset=self.dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,
        )
        train_out = trainer.train()
        metrics = dict(train_out.metrics) if train_out and train_out.metrics else {}
        if self.eval_dataset is not None:
            metrics.update(trainer.evaluate())
        self.model = trainer.model
        return TrainResult(metrics=metrics, checkpoint_dir=out_dir)
