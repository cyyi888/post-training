from __future__ import annotations

from typing import Any

from tqdm import tqdm

from src.algorithms.reward_functions import reward_accuracy, reward_format
from src.core.base import BaseEvaluator
from src.utils.generation import generate_response


class GSM8KEvaluator(BaseEvaluator):
    def evaluate(self, dataset: Any) -> dict[str, float]:
        self.model.eval()
        preds, labels = [], []
        system_prompt = None
        max_new_tokens = int(self.cfg.get("max_new_tokens", 512))

        for example in tqdm(dataset, desc="GSM8K eval"):
            prompt = example["prompt"]
            user_q = next(m["content"] for m in prompt if m["role"] == "user")
            system_prompt = next(
                (m["content"] for m in prompt if m["role"] == "system"), None
            )
            response = generate_response(
                self.model,
                self.tokenizer,
                user_q,
                system_message=system_prompt,
                max_new_tokens=max_new_tokens,
            )
            preds.append([{"role": "assistant", "content": response}])
            labels.append(example["ground_truth"])

        acc = reward_accuracy(preds, labels)
        fmt = reward_format(preds)
        n = max(len(acc), 1)
        return {
            "accuracy": sum(acc) / n,
            "format_rate": sum(fmt) / n,
            "num_samples": float(n),
        }
