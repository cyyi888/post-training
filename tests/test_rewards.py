"""【设备：CPU】GRPO 可插拔奖励引擎与组内优势单测。

原因：校验 boxed/####、format/length/steps 分层，以及 code 测例执行（子进程），
纯 Python + 小张量，不加载基座、不做 generate，无需 CUDA。
"""

from __future__ import annotations

import torch

from src.algorithms.grpo_utils import group_relative_advantages
from src.algorithms.reward_functions import (
    RewardEngine,
    extract_answer,
    extract_python_code,
    has_answer_format,
    reward_accuracy,
    reward_code,
    reward_format,
    reward_length,
    reward_steps,
    score_math_steps,
)


def test_extract_answer_boxed_and_hash():
    assert extract_answer(r"Answer is \boxed{42}.") == "42"
    assert extract_answer("reasoning\n#### 42") == "42"
    assert extract_answer("no answer") == ""
    assert has_answer_format("#### 1")
    assert has_answer_format(r"\boxed{1}")
    assert not has_answer_format("plain")


def test_reward_accuracy_boxed():
    comps = [[{"role": "assistant", "content": r"Answer is \boxed{42}."}]]
    assert reward_accuracy(comps, ["42"]) == [1.0]
    assert reward_accuracy(comps, ["0"]) == [0.0]


def test_reward_accuracy_hash_gsm8k():
    comps = [[{"role": "assistant", "content": "step by step\n#### 7"}]]
    assert reward_accuracy(comps, ["7"]) == [1.0]
    assert reward_format(comps) == [1.0]


def test_reward_format():
    ok = [[{"role": "assistant", "content": r"\boxed{1}"}]]
    bad = [[{"role": "assistant", "content": "no box"}]]
    assert reward_format(ok) == [1.0]
    assert reward_format(bad) == [0.0]


def test_reward_length_layered():
    lr = {"min_tokens": 10, "max_tokens": 100, "target_min": 20, "target_max": 50}
    comps = [[{"role": "assistant", "content": "x"}] for _ in range(4)]
    scores = reward_length(
        comps,
        completion_lengths=[5, 20, 40, 120],
        length_reward=lr,
    )
    assert scores[0] == 0.0  # 过短
    assert scores[1] == 1.0
    assert scores[2] == 1.0
    assert scores[3] == 0.0  # 过长


def test_reward_length_interpolation():
    lr = {"min_tokens": 0, "max_tokens": 100, "target_min": 40, "target_max": 60}
    comps = [[{"role": "assistant", "content": "x"}]]
    scores = reward_length(comps, completion_lengths=[20], length_reward=lr)
    assert abs(scores[0] - 0.5) < 1e-6


def test_reward_engine_weighted():
    engine = RewardEngine(["accuracy", "format"], weights=[1.0, 0.5])
    comps = [[{"role": "assistant", "content": r"\boxed{7}"}]]
    scores = engine(comps, ground_truth=["7"])
    assert scores == [1.5]


def test_reward_engine_score_detailed():
    engine = RewardEngine(
        ["accuracy", "format", "length"],
        weights=[1.0, 0.2, 0.1],
        length_reward={"min_tokens": 1, "max_tokens": 100, "target_min": 1, "target_max": 100},
    )
    comps = [[{"role": "assistant", "content": r"step\boxed{3}"}]]
    detailed = engine.score_detailed(
        comps,
        ground_truth=["3"],
        completion_lengths=[10],
    )
    assert detailed["parts"]["accuracy"] == [1.0]
    assert detailed["parts"]["format"] == [1.0]
    assert detailed["parts"]["length"] == [1.0]
    assert detailed["parts"]["reward_correct"] == [1.0]
    assert abs(detailed["total"][0] - 1.3) < 1e-6
    assert engine(comps, ground_truth=["3"], completion_lengths=[10]) == detailed["total"]


def test_reward_steps_prefers_structured_reasoning():
    poor = "42"
    rich = (
        "Step 1: compute 20+20\n"
        "20 + 20 = 40\n"
        "Step 2: add 2\n"
        "40 + 2 = 42\n"
        r"Final: \boxed{42}"
    )
    s_poor = score_math_steps(poor, "42")
    s_rich = score_math_steps(rich, "42")
    assert s_rich > s_poor
    assert s_rich > 0.7
    scores = reward_steps(
        [[{"role": "assistant", "content": poor}], [{"role": "assistant", "content": rich}]],
        ground_truth=["42", "42"],
    )
    assert scores[1] > scores[0]


def test_extract_python_code_from_fence():
    text = "Here is the solution:\n```python\ndef add(a, b):\n    return a + b\n```\n"
    code = extract_python_code(text)
    assert "def add" in code and "return a + b" in code


def test_reward_code_compile_and_pass_tests():
    prompt = "def add(a, b):\n"
    completion = "    return a + b\n"
    test = (
        "def check(candidate):\n"
        "    assert candidate(1, 2) == 3\n"
        "    assert candidate(0, 0) == 0\n"
    )
    ok = reward_code(
        [[{"role": "assistant", "content": completion}]],
        prompts=[prompt],
        tests=[test],
        entry_points=["add"],
        code_reward={"timeout": 2.0},
    )
    assert ok == [1.0]

    bad = reward_code(
        [[{"role": "assistant", "content": "    return a - b\n"}]],
        prompts=[prompt],
        tests=[test],
        entry_points=["add"],
    )
    assert bad[0] == 0.3  # 可运行但测例失败

    syn = reward_code(
        [[{"role": "assistant", "content": "    return a +\n"}]],
        prompts=[prompt],
        tests=[test],
        entry_points=["add"],
    )
    assert syn[0] == 0.0


def test_reward_engine_with_steps_and_code_aliases():
    engine = RewardEngine(["steps", "code"], weights=[0.5, 0.5])
    text = "Step 1: 1+1=2\n#### 2"
    prompt = "def add(a, b):\n"
    completion = "    return a + b\n"
    test = "def check(candidate):\n    assert candidate(2, 3) == 5\n"
    detailed = engine.score_detailed(
        [
            [{"role": "assistant", "content": text}],
            [{"role": "assistant", "content": completion}],
        ],
        ground_truth=["2", None],
        prompts=["", prompt],
        tests=[None, test],
        entry_points=[None, "add"],
    )
    assert "reward_steps" in detailed["parts"]
    assert "reward_code" in detailed["parts"]
    assert detailed["parts"]["steps"][0] > 0.0
    assert detailed["parts"]["code"][1] == 1.0


def test_group_relative_advantages_zero_mean():
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.5, 0.5])
    adv = group_relative_advantages(rewards, num_generations=2)
    grouped = adv.view(3, 2)
    assert torch.allclose(grouped.mean(dim=1), torch.zeros(3), atol=1e-5)
