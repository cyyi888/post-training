"""【设备：CPU】MetricSink 验收：去重路径、rank、HF 映射、目录。"""

from __future__ import annotations

from types import SimpleNamespace

from src.utils.metric_catalog import METRIC_CATALOG, catalog_keys_for
from src.utils.metric_sink import (
    STEP_OPTIMIZER,
    MetricSink,
    filter_scalar_metrics,
    map_hf_trainer_logs,
)
from src.utils.wandb_utils import resolve_hf_report_to, resolve_report_to


def test_hf_report_to_always_none_to_avoid_double_log():
    assert resolve_hf_report_to({"wandb": {"enabled": True, "mode": "online"}}) == "none"
    assert resolve_report_to(
        {"wandb": {"enabled": True, "mode": "online"}, "report_to": "none"}
    ) == "wandb"


def test_metric_sink_rank0_only():
    captured: list = []

    def backend(metrics, step=None):
        captured.append((step, dict(metrics)))

    sink_main = MetricSink(
        enabled=True, is_main_process=True, algorithm="sft", backend_log=backend
    )
    sink_worker = MetricSink(
        enabled=True, is_main_process=False, algorithm="sft", backend_log=backend
    )
    sink_main.log({"train/loss": 1.0}, step=3, step_kind=STEP_OPTIMIZER)
    sink_worker.log({"train/loss": 2.0}, step=3, step_kind=STEP_OPTIMIZER)
    assert len(captured) == 1
    assert captured[0][0] == 3
    assert captured[0][1]["train/loss"] == 1.0
    assert captured[0][1]["meta/step_kind"] == STEP_OPTIMIZER
    assert captured[0][1]["meta/algorithm"] == "sft"


def test_metric_sink_disabled_backend_still_records_history():
    sink = MetricSink(
        enabled=False, is_main_process=True, algorithm="dpo", record_history=True
    )
    sink.log({"train/mean_kl": 0.5}, step=1, step_kind=STEP_OPTIMIZER)
    assert len(sink.history) == 1
    assert sink.history[0]["metrics"]["train/mean_kl"] == 0.5


def test_map_hf_trainer_logs_public_keys():
    mapped = map_hf_trainer_logs({"loss": 0.2, "learning_rate": 1e-5, "epoch": 0.5, "weird": 1})
    assert mapped["train/loss"] == 0.2
    assert mapped["train/learning_rate"] == 1e-5
    assert mapped["train/epoch"] == 0.5
    assert mapped["hf/weird"] == 1


def test_filter_scalar_metrics():
    out = filter_scalar_metrics({"a": 1, "b": [1, 2], "c": "x"})
    assert out == {"a": 1, "c": "x"}


def test_catalog_separates_dpo_kl_and_grpo_kl():
    assert "train/mean_kl" in METRIC_CATALOG
    assert "train/kl" in METRIC_CATALOG
    assert "dpo" in METRIC_CATALOG["train/mean_kl"]["algorithms"]
    assert "grpo" in METRIC_CATALOG["train/kl"]["algorithms"]
    assert "grpo" not in METRIC_CATALOG["train/mean_kl"]["algorithms"]
    dpo_keys = catalog_keys_for("dpo")
    assert "train/mean_kl" in dpo_keys
    assert "train/kl" not in dpo_keys


def test_hf_callback_forwards_to_sink():
    pytest = __import__("pytest")
    pytest.importorskip("transformers")
    from src.utils.metric_sink import make_metric_sink_callback

    sink = MetricSink(
        enabled=False, is_main_process=True, algorithm="sft", record_history=True
    )
    cb = make_metric_sink_callback(sink, algorithm="sft")
    state = SimpleNamespace(global_step=7)
    cb.on_log(None, state, None, logs={"loss": 0.3, "learning_rate": 1e-4})
    assert len(sink.history) == 1
    assert sink.history[0]["step"] == 7
    assert sink.history[0]["metrics"]["train/loss"] == 0.3
    assert sink.history[0]["step_kind"] == STEP_OPTIMIZER
