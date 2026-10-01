"""DPO / IPO 损失与动态 β 单元测试（纯 CPU）。"""

from __future__ import annotations

import torch

from src.algorithms.dpo_loss import DPOLoss, DPOLossConfig, compute_dpo_loss
from src.algorithms.dpo_losses import dpo_loss, ipo_loss
from src.algorithms.dpo_variants import compute_ipo_loss, compute_variant_loss
from src.algorithms.dynamic_beta import DynamicBetaController


def _logps(n: int = 8, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    # chosen 略优于 rejected
    policy_chosen = torch.randn(n, generator=g) * 0.5 + 1.0
    policy_rejected = torch.randn(n, generator=g) * 0.5 - 0.5
    ref_chosen = torch.randn(n, generator=g) * 0.3
    ref_rejected = torch.randn(n, generator=g) * 0.3
    return policy_chosen, policy_rejected, ref_chosen, ref_rejected


def test_dpo_loss_positive_and_decreases_with_margin():
    pc, pr, rc, rr = _logps(seed=1)
    loss_small, _ = dpo_loss(pc, pr, rc, rr, beta=0.1)

    # 拉大 chosen / rejected 差距 → 损失应更小
    pc2 = pc + 2.0
    pr2 = pr - 2.0
    loss_large, metrics = dpo_loss(pc2, pr2, rc, rr, beta=0.1)

    assert torch.isfinite(loss_small)
    assert torch.isfinite(loss_large)
    assert float(loss_small) > 0.0
    assert float(loss_large) < float(loss_small)
    assert "mean_kl" in metrics and "logps_diff" in metrics


def test_ipo_hinge_zero_when_margin_satisfied():
    # 构造 delta 很大，使 beta*delta >> margin → hinge = 0
    pc = torch.tensor([10.0, 10.0])
    pr = torch.tensor([-10.0, -10.0])
    rc = torch.zeros(2)
    rr = torch.zeros(2)
    loss, metrics = ipo_loss(pc, pr, rc, rr, beta=0.1, margin=1.0)
    assert float(loss.item()) == 0.0
    assert metrics["loss"] == 0.0


def test_ipo_hinge_positive_when_below_margin():
    pc = torch.tensor([0.0, 0.0])
    pr = torch.tensor([0.0, 0.0])
    rc = torch.zeros(2)
    rr = torch.zeros(2)
    loss, _ = ipo_loss(pc, pr, rc, rr, beta=0.1, margin=1.0)
    # max(0, 1 - 0) = 1
    assert abs(float(loss.item()) - 1.0) < 1e-5


def test_dynamic_beta_adjustments_and_clip():
    ctl = DynamicBetaController(
        init_beta=0.1, target_kl=4.0, adjust_rate=0.1, min_beta=0.01, max_beta=1.0
    )
    # KL 过高 → beta 增大
    b1 = ctl.step(10.0)
    assert b1 > 0.1

    ctl2 = DynamicBetaController(init_beta=0.1, target_kl=4.0, adjust_rate=0.1)
    # KL 过低 → beta 减小
    b2 = ctl2.step(1.0)
    assert b2 < 0.1

    ctl3 = DynamicBetaController(init_beta=0.2, target_kl=4.0, adjust_rate=0.1)
    # 落在 [0.9, 1.1] * target → 不变
    b3 = ctl3.step(4.0)
    assert abs(b3 - 0.2) < 1e-9

    # 上限裁剪
    ctl4 = DynamicBetaController(
        init_beta=0.95, target_kl=1.0, adjust_rate=0.5, min_beta=0.01, max_beta=1.0
    )
    b4 = ctl4.step(100.0)
    assert b4 <= 1.0

    # 下限裁剪
    ctl5 = DynamicBetaController(
        init_beta=0.02, target_kl=10.0, adjust_rate=0.5, min_beta=0.01, max_beta=1.0
    )
    b5 = ctl5.step(0.01)
    assert b5 >= 0.01


def test_dynamic_beta_state_roundtrip():
    ctl = DynamicBetaController(init_beta=0.2, target_kl=3.0)
    ctl.step(9.0)
    state = ctl.state_dict()
    restored = DynamicBetaController()
    restored.load_state_dict(state)
    assert abs(restored.beta - ctl.beta) < 1e-9
    assert restored.steps == ctl.steps


def test_numerical_stability_extreme_gaps():
    pc = torch.tensor([1e4, -1e4])
    pr = torch.tensor([-1e4, 1e4])
    rc = torch.zeros(2)
    rr = torch.zeros(2)
    loss_dpo, m1 = dpo_loss(pc, pr, rc, rr, beta=0.1)
    loss_ipo, m2 = ipo_loss(pc, pr, rc, rr, beta=0.1, margin=1.0)
    assert torch.isfinite(loss_dpo) and not torch.isnan(loss_dpo)
    assert torch.isfinite(loss_ipo) and not torch.isnan(loss_ipo)
    assert all(torch.isfinite(torch.tensor(v)) for v in m1.values() if isinstance(v, float))
    assert all(torch.isfinite(torch.tensor(v)) for v in m2.values() if isinstance(v, float))


def test_variant_dispatch_and_module():
    pc, pr, rc, rr = _logps(seed=4)
    loss_a, _ = compute_variant_loss("dpo", pc, pr, rc, rr, DPOLossConfig(beta=0.2))
    loss_b, _ = compute_dpo_loss(pc, pr, rc, rr, DPOLossConfig(beta=0.2))
    assert abs(float(loss_a) - float(loss_b)) < 1e-6

    loss_c, _ = compute_ipo_loss(pc, pr, rc, rr, DPOLossConfig(loss_type="ipo", margin=0.5))
    assert torch.isfinite(loss_c)

    module = DPOLoss(DPOLossConfig(loss_type="dpo", beta=0.1))
    loss_d, metrics = module(pc, pr, rc, rr)
    assert torch.isfinite(loss_d)
    assert "mean_kl" in metrics


def test_loss_config_beta_mode_compat():
    cfg = DPOLossConfig(dynamic_beta=True, beta_mode="fixed")
    assert cfg.resolved_beta_mode() == "dynamic"
    cfg2 = DPOLossConfig(beta_mode="dynamic")
    assert cfg2.resolved_beta_mode() == "dynamic"
    cfg3 = DPOLossConfig()
    assert cfg3.loss_type == "dpo"
    assert cfg3.resolved_beta_mode() == "fixed"
