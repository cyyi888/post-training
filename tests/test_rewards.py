"""Tests for GRPO reward engine and advantage helpers."""

from __future__ import annotations

import torch

from src.algorithms.grpo_utils import group_relative_advantages
from src.algorithms.reward_functions import RewardEngine, reward_accuracy, reward_format


def test_reward_accuracy_boxed():
    comps = [[{"role": "assistant", "content": r"Answer is \boxed{42}."}]]
    assert reward_accuracy(comps, ["42"]) == [1.0]
    assert reward_accuracy(comps, ["0"]) == [0.0]


def test_reward_format():
    ok = [[{"role": "assistant", "content": r"\boxed{1}"}]]
    bad = [[{"role": "assistant", "content": "no box"}]]
    assert reward_format(ok) == [1.0]
    assert reward_format(bad) == [0.0]


def test_reward_engine_weighted():
    engine = RewardEngine(["accuracy", "format"], weights=[1.0, 0.5])
    comps = [[{"role": "assistant", "content": r"\boxed{7}"}]]
    scores = engine(comps, ground_truth=["7"])
    assert scores == [1.5]


def test_group_relative_advantages_zero_mean():
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.5, 0.5])
    adv = group_relative_advantages(rewards, num_generations=2)
    grouped = adv.view(3, 2)
    assert torch.allclose(grouped.mean(dim=1), torch.zeros(3), atol=1e-5)
