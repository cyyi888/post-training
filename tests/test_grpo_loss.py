"""自研 GRPO loss 的数值测试（只需 torch）。"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.algorithms.grpo_loss import advantages_from_rewards, compute_grpo_loss
from src.algorithms.grpo_utils import group_relative_advantages


def _pack(n: int = 4, t: int = 3):
    policy = torch.zeros(n, t, requires_grad=True)
    old = torch.zeros(n, t)
    mask = torch.ones(n, t)
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    adv = advantages_from_rewards(rewards, num_generations=2)
    return policy, old, mask, adv


def test_equal_ratio_zero_mean_advantage_loss_near_zero():
    policy, old, mask, adv = _pack()
    assert torch.allclose(adv.view(2, 2).mean(dim=1), torch.zeros(2), atol=1e-5)
    loss, metrics = compute_grpo_loss(policy, old, adv, mask, ref_logps=None, beta=0.0)
    # ρ=1 时 loss = -mean(advantage)，组内优势均值约为 0
    assert abs(metrics["loss"]) < 1e-5
    assert loss.ndim == 0


def test_positive_advantage_prefers_higher_logprob():
    """正优势样本上提高当前策略 logprob，loss 应下降。"""
    adv = torch.tensor([1.0, 1.0])
    mask = torch.ones(2, 2)
    old = torch.zeros(2, 2)
    low = torch.zeros(2, 2, requires_grad=True)
    high = torch.full((2, 2), 0.5, requires_grad=True)
    loss_low, _ = compute_grpo_loss(low, old, adv, mask, beta=0.0)
    loss_high, _ = compute_grpo_loss(high, old, adv, mask, beta=0.0)
    assert loss_high < loss_low


def test_kl_increases_loss_when_policy_drifts():
    policy = torch.zeros(2, 2, requires_grad=True)
    old = torch.zeros(2, 2)
    ref = torch.full((2, 2), -1.0)  # 参考与当前差一截
    adv = torch.zeros(2)
    mask = torch.ones(2, 2)
    no_kl, m0 = compute_grpo_loss(policy, old, adv, mask, ref_logps=ref, beta=0.0)
    with_kl, m1 = compute_grpo_loss(policy, old, adv, mask, ref_logps=ref, beta=0.1)
    assert with_kl > no_kl
    assert m1["policy/kl"] > 0
    assert m0["policy/kl"] == 0.0


def test_clip_limits_ratio():
    # log ρ = log(3) → ρ=3，clip ε=0.2 后 ρ 被夹到 1.2
    policy = torch.full((2, 1), 1.0986123, requires_grad=True)  # ln(3)
    old = torch.zeros(2, 1)
    adv = torch.tensor([1.0, 1.0])
    mask = torch.ones(2, 1)
    _, metrics = compute_grpo_loss(policy, old, adv, mask, beta=0.0, clip_range=0.2)
    assert metrics["policy/clipfrac"] > 0.9


def test_group_advantages_g1_is_zero():
    rewards = torch.tensor([1.0, 2.0, 3.0])
    adv = group_relative_advantages(rewards, num_generations=1)
    assert torch.allclose(adv, torch.zeros_like(rewards))


def test_loss_backward():
    policy, old, mask, adv = _pack()
    loss, _ = compute_grpo_loss(policy, old, adv, mask, beta=0.0)
    loss.backward()
    assert policy.grad is not None
