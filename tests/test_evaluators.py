"""【设备：CPU】任务评测器（GSM8K / HumanEval）单测。

原因：用 mock 生成，不加载基座、不访问 HF；校验指标聚合与 pass@k。
"""

from __future__ import annotations

from types import SimpleNamespace

from datasets import Dataset

from src.evaluators import available_tasks, build_evaluator
from src.evaluators.common import pass_at_k
from src.evaluators.gsm8k_eval import GSM8KEvaluator
from src.evaluators.humaneval_eval import HumanEvalEvaluator


class _DummyModel:
    def eval(self):
        return self


class _DummyTok:
    pass


def test_pass_at_k_edges():
    assert pass_at_k(1, 1, 1) == 1.0
    assert pass_at_k(1, 0, 1) == 0.0
    assert pass_at_k(5, 5, 1) == 1.0
    assert pass_at_k(5, 0, 1) == 0.0
    # n=5, c=1, k=1 → 1/5
    assert abs(pass_at_k(5, 1, 1) - 0.2) < 1e-9
    # 全部通过 → 1
    assert pass_at_k(10, 10, 10) == 1.0


def test_available_tasks_registered():
    tasks = available_tasks()
    assert "gsm8k" in tasks
    assert "humaneval" in tasks


def test_build_evaluator_aliases():
    cfg = {"eval": {"task": "gsm8k", "max_new_tokens": 32}}
    ev = build_evaluator("gsm8k", cfg, _DummyModel(), _DummyTok())
    assert isinstance(ev, GSM8KEvaluator)
    ev2 = build_evaluator("human_eval", cfg, _DummyModel(), _DummyTok())
    assert isinstance(ev2, HumanEvalEvaluator)


def test_gsm8k_evaluator_metrics(monkeypatch):
    answers = [
        r"step\n\boxed{42}",
        "no format here",
        "#### 7",
    ]

    def _fake_gen(*_a, **_k):
        return answers.pop(0)

    monkeypatch.setattr("src.evaluators.gsm8k_eval.generate_completion", _fake_gen)

    ds = Dataset.from_list(
        [
            {
                "prompt": [{"role": "user", "content": "q1"}],
                "ground_truth": "42",
            },
            {
                "prompt": [{"role": "user", "content": "q2"}],
                "ground_truth": "1",
            },
            {
                "prompt": [{"role": "user", "content": "q3"}],
                "ground_truth": "7",
            },
        ]
    )
    cfg = {"eval": {"max_new_tokens": 64}}
    metrics = GSM8KEvaluator(cfg, _DummyModel(), _DummyTok()).evaluate(ds)
    assert metrics["num_samples"] == 3.0
    assert abs(metrics["accuracy"] - (2 / 3)) < 1e-9
    assert abs(metrics["format_rate"] - (2 / 3)) < 1e-9


def test_humaneval_evaluator_pass_rate(monkeypatch):
    # 1) 正确实现  2) 语法错误  3) 逻辑错误
    completions = [
        "    return a + b\n",
        "    return a +\n",
        "    return a - b\n",
    ]

    def _fake_gen(*_a, **_k):
        return completions.pop(0)

    monkeypatch.setattr("src.evaluators.humaneval_eval.generate_completion", _fake_gen)

    test = (
        "def check(candidate):\n"
        "    assert candidate(1, 2) == 3\n"
        "    assert candidate(0, 0) == 0\n"
    )
    ds = Dataset.from_list(
        [
            {
                "prompt": [{"role": "user", "content": "def add(a, b):\n"}],
                "task_prompt": "def add(a, b):\n",
                "test": test,
                "entry_point": "add",
                "ground_truth": "    return a + b\n",
            },
            {
                "prompt": [{"role": "user", "content": "def add(a, b):\n"}],
                "task_prompt": "def add(a, b):\n",
                "test": test,
                "entry_point": "add",
            },
            {
                "prompt": [{"role": "user", "content": "def add(a, b):\n"}],
                "task_prompt": "def add(a, b):\n",
                "test": test,
                "entry_point": "add",
            },
        ]
    )
    cfg = SimpleNamespace(
        eval={
            "max_new_tokens": 64,
            "num_samples": 1,
            "k": 1,
            "timeout": 2.0,
            "temperature": 0.0,
        }
    )
    # DictConfig-like via plain dict for eval_section
    cfg = {"eval": dict(cfg.eval)}
    metrics = HumanEvalEvaluator(cfg, _DummyModel(), _DummyTok()).evaluate(ds)
    assert metrics["num_problems"] == 3.0
    assert abs(metrics["pass_at_1"] - (1 / 3)) < 1e-9
    assert abs(metrics["pass_rate"] - metrics["pass_at_1"]) < 1e-9
    # 语法错误不计 compile；通过 + 逻辑错误 = 2/3
    assert abs(metrics["compile_rate"] - (2 / 3)) < 1e-9


def test_humaneval_pass_at_k_with_multi_sample(monkeypatch):
    # 同一题采 2 次：先错后对 → pass@1=0.5, pass@2=1
    seq = ["    return a - b\n", "    return a + b\n"]

    def _fake_gen(*_a, **_k):
        return seq.pop(0)

    monkeypatch.setattr("src.evaluators.humaneval_eval.generate_completion", _fake_gen)

    test = "def check(candidate):\n    assert candidate(2, 3) == 5\n"
    ds = Dataset.from_list(
        [
            {
                "prompt": [{"role": "user", "content": "def add(a, b):\n"}],
                "task_prompt": "def add(a, b):\n",
                "test": test,
                "entry_point": "add",
            }
        ]
    )
    cfg = {
        "eval": {
            "max_new_tokens": 32,
            "num_samples": 2,
            "k": 2,
            "timeout": 2.0,
            "temperature": 0.2,
        }
    }
    metrics = HumanEvalEvaluator(cfg, _DummyModel(), _DummyTok()).evaluate(ds)
    assert abs(metrics["pass_at_1"] - 0.5) < 1e-9
    assert abs(metrics["pass_at_2"] - 1.0) < 1e-9
