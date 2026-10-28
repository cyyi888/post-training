"""【设备：CPU】DPO / IPO 损失与动态 β 公式单测。

原因：只验证偏好损失、合页与动态 β 的数值行为，小张量即可，不加载基座、无需 CUDA。
端到端 TRL 短训见 ``tests/test_dpo_pipeline.py``（GPU）。
"""

from __future__ import annotations

import pytest
import torch

from src.algorithms.dpo_losses import (
    DPOLossConfig,
    compute_preference_loss,
    dpo_loss,
    ipo_loss,
)
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


def test_preference_loss_dispatch():
    pc, pr, rc, rr = _logps(seed=4)
    loss_dpo, metrics = compute_preference_loss(
        "dpo", pc, pr, rc, rr, beta=0.2
    )
    loss_direct, _ = dpo_loss(pc, pr, rc, rr, beta=0.2)
    assert abs(float(loss_dpo) - float(loss_direct)) < 1e-6
    assert "mean_kl" in metrics

    loss_ipo, _ = compute_preference_loss(
        "ipo", pc, pr, rc, rr, beta=0.1, margin=0.5
    )
    assert torch.isfinite(loss_ipo)

    with pytest.raises(ValueError, match="未知 loss_type"):
        compute_preference_loss("unknown", pc, pr, rc, rr, beta=0.1)


def test_loss_config_beta_mode():
    cfg = DPOLossConfig(beta_mode="dynamic")
    assert cfg.resolved_beta_mode() == "dynamic"
    cfg2 = DPOLossConfig(beta_mode="fixed")
    assert cfg2.resolved_beta_mode() == "fixed"
    cfg3 = DPOLossConfig()
    assert cfg3.loss_type == "dpo"
    assert cfg3.resolved_beta_mode() == "fixed"
    try:
        DPOLossConfig(beta_mode="auto").resolved_beta_mode()
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "fixed|dynamic" in str(exc)


def test_training_cfg_beta_api_unified():
    """Hydra training 只认 beta_mode / fixed_beta / dynamic_beta 表。"""
    from omegaconf import OmegaConf

    from src.trainers.dpo_trainer import _build_loss_cfg, _resolve_beta_mode, _resolve_fixed_beta

    tcfg = OmegaConf.create(
        {
            "beta_mode": "fixed",
            "fixed_beta": 0.3,
            "dynamic_beta": {"init_beta": 0.1, "target_kl": 4.0},
            "beta": 0.99,  # 旧字段：应被忽略
        }
    )
    assert _resolve_beta_mode(tcfg) == "fixed"
    assert abs(_resolve_fixed_beta(tcfg) - 0.3) < 1e-9
    cfg = _build_loss_cfg(tcfg)
    assert abs(cfg.beta - 0.3) < 1e-9

    with pytest.raises(ValueError, match="嵌套表"):
        _build_loss_cfg(OmegaConf.create({"beta_mode": "dynamic", "dynamic_beta": True}))
