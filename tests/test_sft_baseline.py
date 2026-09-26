"""SFT 自研采样器与梯度裁剪单测（不依赖 GPU / TRL）。"""

from __future__ import annotations

import pytest

from src.algorithms.sft_samplers import (
    BucketSampler,
    LengthGroupedSampler,
    RandomOrderSampler,
    build_sft_sampler,
    estimate_example_length,
    length_group_waste_ratio,
)


def test_estimate_example_length_messages():
    ex = {
        "messages": [
            {"role": "user", "content": "hello " * 20},
            {"role": "assistant", "content": "world " * 20},
        ]
    }
    n = estimate_example_length(ex, max_length=512)
    assert 1 <= n <= 512


def test_random_sampler_reproducible():
    a = list(RandomOrderSampler(20, seed=3, shuffle=True))
    b = list(RandomOrderSampler(20, seed=3, shuffle=True))
    assert a == b
    assert sorted(a) == list(range(20))


def test_length_group_reduces_padding_waste():
    lengths = [10, 200, 12, 190, 11, 185, 15, 195, 9, 180, 14, 170]
    grouped = LengthGroupedSampler(lengths, batch_size=2, mega_batch_mult=6, seed=0)
    order = list(grouped)
    assert sorted(order) == list(range(len(lengths)))
    waste_group = length_group_waste_ratio(lengths, order, batch_size=2)

    random_order = list(RandomOrderSampler(len(lengths), seed=0, shuffle=True))
    waste_random = length_group_waste_ratio(lengths, random_order, batch_size=2)
    assert waste_group <= waste_random


def test_bucket_sampler_covers_all():
    lengths = [50, 100, 200, 400, 800, 30, 90]
    sampler = BucketSampler(
        lengths, batch_size=2, boundaries=[64, 128, 256, 512], seed=1, drop_last=False
    )
    order = list(sampler)
    assert sorted(order) == list(range(len(lengths)))


def test_build_sft_sampler_modes():
    data = [{"messages": [{"role": "user", "content": "x" * n}]} for n in (20, 80, 200, 40)]
    s = build_sft_sampler(data, mode="length_group", batch_size=2, seed=0)
    assert len(list(s)) == 4
    s2 = build_sft_sampler(data, mode="bucket", batch_size=2, boundaries=[64, 256], seed=0)
    assert len(list(s2)) == 4
    s3 = build_sft_sampler(data, mode="random", batch_size=2, seed=0)
    assert len(list(s3)) == 4


def test_grad_clip_norm_and_value():
    torch = pytest.importorskip("torch")
    from src.algorithms.grad_clip import GradClipConfig, GradClipper

    model = torch.nn.Linear(4, 4, bias=False)
    loss = model(torch.ones(2, 4)).sum()
    loss.backward()
    raw = GradClipper(GradClipConfig(strategy="none")).clip(model)

    # 重新反传
    model.zero_grad()
    model(torch.ones(2, 4)).sum().backward()
    clipper = GradClipper(GradClipConfig(strategy="norm", max_norm=0.05))
    clipped = clipper.clip(model)
    assert clipped <= 0.05 + 1e-5
    assert clipper.last_clip_coef <= 1.0

    model.zero_grad()
    model(torch.ones(2, 4)).sum().backward()
    vclip = GradClipper(GradClipConfig(strategy="value", max_value=0.01))
    vclip.clip(model)
    for p in model.parameters():
        assert float(p.grad.abs().max()) <= 0.01 + 1e-6


def test_grad_clip_adaptive_updates_history():
    torch = pytest.importorskip("torch")
    from src.algorithms.grad_clip import GradClipConfig, GradClipper

    model = torch.nn.Linear(2, 2, bias=False)
    clipper = GradClipper(
        GradClipConfig(strategy="adaptive", max_norm=10.0, adaptive_window=5, adaptive_min_norm=0.01)
    )
    for _ in range(3):
        model.zero_grad()
        model(torch.randn(3, 2)).sum().backward()
        clipper.clip(model)
    assert len(clipper._history) == 3
    assert "grad/raw_norm" in clipper.metrics()


def test_sft_config_sections_parse():
    from src.trainers.sft_trainer import _as_dict

    class Sec:
        def __init__(self, **kw):
            self.__dict__.update(kw)

        def get(self, k, d=None):
            return self.__dict__.get(k, d)

    d = _as_dict(Sec(mode="length_group", mega_batch_mult=4))
    assert d["mode"] == "length_group"
    assert d["mega_batch_mult"] == 4
