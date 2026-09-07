from __future__ import annotations

from typing import Any

from trl import GRPOConfig, GRPOTrainer

from src.algorithms.reward_functions import build_reward_funcs
from src.core.base import BaseTrainer, TrainResult
from src.core.registry import ALGORITHM_REGISTRY, register


@register(ALGORITHM_REGISTRY, "grpo")
class GRPOStageTrainer(BaseTrainer):
    def __init__(self, cfg: Any, model: Any, tokenizer: Any, dataset: Any):
        super().__init__(cfg, model, tokenizer, dataset)
        self.reward_fn = build_reward_funcs(cfg.training.reward)

    def train(self) -> TrainResult:
        tcfg = self.cfg.training
        out_dir = f"{self.cfg.output_dir}/{tcfg.output_subdir}"
        args = GRPOConfig(
            output_dir=out_dir,
            per_device_train_batch_size=tcfg.per_device_train_batch_size,
            gradient_accumulation_steps=tcfg.gradient_accumulation_steps,
            num_generations=tcfg.num_generations,
            num_train_epochs=tcfg.num_train_epochs,
            max_completion_length=tcfg.max_completion_length,
            learning_rate=tcfg.learning_rate,
            logging_steps=tcfg.logging_steps,
            seed=self.cfg.seed,
            report_to=self.cfg.get("report_to", "none"),
            save_strategy=tcfg.get("save_strategy", "no"),
        )
        trainer = GRPOTrainer(
            model=self.model,
            args=args,
            train_dataset=self.dataset,
            reward_funcs=self.reward_fn,
            processing_class=self.tokenizer,
        )
        train_out = trainer.train()
        metrics = dict(train_out.metrics) if train_out and train_out.metrics else {}
        self.model = trainer.model
        return TrainResult(metrics=metrics, checkpoint_dir=out_dir)
