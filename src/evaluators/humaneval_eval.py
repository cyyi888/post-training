"""HumanEval 自动评测：测例执行 pass@k。"""

from __future__ import annotations

from typing import Any

from tqdm import tqdm

from src.algorithms.reward_functions import run_humaneval_tests
from src.core.base import BaseEvaluator
from src.core.registry import EVALUATOR_REGISTRY, register
from src.evaluators.common import eval_section, generate_completion, pass_at_k, user_and_system


@register(EVALUATOR_REGISTRY, "humaneval")
class HumanEvalEvaluator(BaseEvaluator):
    """对投影后的 HumanEval 样本生成代码并用官方测例判分。"""

    def evaluate(self, dataset: Any) -> dict[str, float]:
        self.model.eval()
        ecfg = eval_section(self.cfg)
        max_new_tokens = int(ecfg.get("max_new_tokens", 512))
        n_sample = max(int(ecfg.get("num_samples", 1)), 1)
        k = max(int(ecfg.get("k", min(n_sample, 1))), 1)
        if k > n_sample:
            k = n_sample
        timeout = float(ecfg.get("timeout", 3.0))
        temperature = float(ecfg.get("temperature", 0.0 if n_sample == 1 else 0.2))
        top_p = float(ecfg.get("top_p", 0.95))

        pass_ks: list[float] = []
        pass1s: list[float] = []
        compile_hits = 0
        total_gens = 0

        for example in tqdm(dataset, desc="HumanEval eval"):
            user_q, _ = user_and_system(example)
            task_prompt = str(example.get("task_prompt") or user_q or "")
            test = example.get("test")
            entry = example.get("entry_point")
            if not test or not entry:
                raise ValueError(
                    "HumanEval 样本缺少 test / entry_point；请用 algorithm=grpo 投影数据"
                )

            n_pass = 0
            n_compile = 0
            for _ in range(n_sample):
                response = generate_completion(
                    self.model,
                    self.tokenizer,
                    example,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if n_sample > 1 else 0.0,
                    top_p=top_p,
                )
                ok, reason = run_humaneval_tests(
                    task_prompt,
                    response,
                    str(test),
                    str(entry),
                    timeout=timeout,
                )
                total_gens += 1
                if ok:
                    n_pass += 1
                    n_compile += 1
                elif not str(reason).startswith("syntax"):
                    n_compile += 1

            compile_hits += n_compile
            pass1s.append(pass_at_k(n_sample, n_pass, 1))
            pass_ks.append(pass_at_k(n_sample, n_pass, k))

        n = max(len(pass1s), 1)
        metrics: dict[str, float] = {
            "pass_at_1": sum(pass1s) / n,
            f"pass_at_{k}": sum(pass_ks) / n,
            "compile_rate": (compile_hits / max(total_gens, 1)),
            "num_problems": float(len(pass1s)),
            "num_samples": float(n_sample),
        }
        # n=k=1 时 pass_at_1 与 pass_at_k 同义，保留 pass_rate 别名便于日志
        metrics["pass_rate"] = metrics["pass_at_1"]
        return metrics
