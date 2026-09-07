"""Correctness checks for custom DPO / IPO loss."""

from __future__ import annotations

import torch

from src.algorithms.dpo_loss import DPOLoss, DPOLossConfig, compute_dpo_loss


def _rand_logps(n: int = 8, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(n, generator=g),
        torch.randn(n, generator=g),
        torch.randn(n, generator=g),
        torch.randn(n, generator=g),
    )


def test_dpo_loss_finite_and_scalar():
    pc, pr, rc, rr = _rand_logps()
    loss, metrics = compute_dpo_loss(pc, pr, rc, rr, DPOLossConfig(loss_type="dpo", beta=0.1))
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert "rewards/margins" in metrics
    assert metrics["beta"] == 0.1


def test_ipo_loss_differs_from_dpo():
    pc, pr, rc, rr = _rand_logps(seed=1)
    dpo, _ = compute_dpo_loss(pc, pr, rc, rr, DPOLossConfig(loss_type="dpo", beta=0.1))
    ipo, _ = compute_dpo_loss(pc, pr, rc, rr, DPOLossConfig(loss_type="ipo", beta=0.1))
    assert not torch.allclose(dpo, ipo)


def test_preferred_completion_has_lower_loss():
    """When policy strongly prefers chosen, DPO loss should be small."""
    n = 16
    # policy: chosen >> rejected; ref: roughly equal
    policy_chosen = torch.zeros(n)
    policy_rejected = torch.full((n,), -5.0)
    ref_chosen = torch.full((n,), -1.0)
    ref_rejected = torch.full((n,), -1.0)
    loss_good, m_good = compute_dpo_loss(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected,
        DPOLossConfig(beta=0.1),
    )

    # flipped preferences → higher loss
    loss_bad, m_bad = compute_dpo_loss(
        policy_rejected, policy_chosen, ref_chosen, ref_rejected,
        DPOLossConfig(beta=0.1),
    )
    assert loss_good < loss_bad
    assert m_good["rewards/margins"] > m_bad["rewards/margins"]


def test_dynamic_beta_in_range():
    pc, pr, rc, rr = _rand_logps(seed=2)
    cfg = DPOLossConfig(dynamic_beta=True, beta_min=0.05, beta_max=0.5, beta=0.1)
    _, metrics = compute_dpo_loss(pc, pr, rc, rr, cfg)
    assert 0.05 <= metrics["beta"] <= 0.5


def test_length_penalty_increases_loss():
    pc, pr, rc, rr = _rand_logps(seed=3)
    chosen_len = torch.full((8,), 100.0)
    rejected_len = torch.full((8,), 10.0)
    base, _ = compute_dpo_loss(pc, pr, rc, rr, DPOLossConfig(penalty_enabled=False))
    penalized, m = compute_dpo_loss(
        pc, pr, rc, rr,
        DPOLossConfig(penalty_enabled=True, penalty_type="length", penalty_weight=0.1),
        chosen_lengths=chosen_len,
        rejected_lengths=rejected_len,
    )
    assert penalized > base
    assert m["penalty"] > 0


def test_module_forward_matches_function():
    pc, pr, rc, rr = _rand_logps(seed=4)
    cfg = DPOLossConfig(beta=0.2)
    loss_fn, m_fn = compute_dpo_loss(pc, pr, rc, rr, cfg)
    module = DPOLoss(cfg)
    loss_m, m_m = module(pc, pr, rc, rr)
    assert torch.allclose(loss_fn, loss_m)
    assert m_fn["beta"] == m_m["beta"]
