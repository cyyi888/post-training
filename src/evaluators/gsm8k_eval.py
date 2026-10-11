"""GSM8K 自动评测：最终答案准确率 + 答案格式率。"""

from __future__ import annotations

from typing import Any

from tqdm import tqdm

from src.algorithms.reward_functions import reward_accuracy, reward_format
from src.core.base import BaseEvaluator
from src.core.registry import EVALUATOR_REGISTRY, register
from src.evaluators.common import eval_section, generate_completion


@register(EVALUATOR_REGISTRY, "gsm8k")
class GSM8KEvaluator(BaseEvaluator):
    """对投影后的 GSM8K eval split 做贪心生成并打分。"""

    def evaluate(self, dataset: Any) -> dict[str, float]:
        self.model.eval()
        ecfg = eval_section(self.cfg)
        max_new_tokens = int(ecfg.get("max_new_tokens", 512))

        preds: list[list[dict[str, str]]] = []
        labels: list[str] = []

        for example in tqdm(dataset, desc="GSM8K eval"):
            response = generate_completion(
                self.model,
                self.tokenizer,
                example,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
            )
            preds.append([{"role": "assistant", "content": response}])
            labels.append(str(example.get("ground_truth", "") or ""))

        acc = reward_accuracy(preds, labels)
        fmt = reward_format(preds)
        n = max(len(acc), 1)
        return {
            "accuracy": sum(acc) / n,
            "format_rate": sum(fmt) / n,
            "num_samples": float(len(acc)),
        }
