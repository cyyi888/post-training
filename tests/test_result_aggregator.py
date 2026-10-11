"""【设备：CPU】ResultAggregator 专项验收（Fixture = test_fixture，非真实结论）。"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

import pytest

from src.experiment.aggregator import ResultAggregator


def _write_meta(root: Path, **extra):
    meta = {
        "experiment_name": root.name,
        "algorithm": "dpo",
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "dataset": "dpo_demo",
        "sweep": {"training.fixed_beta": [0.1, 0.5]},
        "sweep_parameters": ["training.fixed_beta"],
        "baseline_overrides": {"training.loss_type": "dpo"},
        "seeds": [12, 34],
        "eval_task": "gsm8k",
        "recorded_at": "2026-01-01T00:00:00+00:00",
    }
    meta.update(extra)
    root.mkdir(parents=True, exist_ok=True)
    (root / "experiment_meta.json").write_text(json.dumps(meta), encoding="utf-8")


def _write_run(
    root: Path,
    *,
    run_id: str,
    variant: str,
    seed: int,
    status: str,
    task: str,
    metrics: dict | None,
    error: str | None = None,
    sweep_value: float | None = None,
    checkpoint: str | None = None,
):
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = None
    if metrics is not None:
        report_path = run_dir / "eval_report.json"
        # 阶段 A EvaluationResult 形状
        report_path.write_text(
            json.dumps(
                {
                    "task": task,
                    "checkpoint": checkpoint or str(run_dir / "ckpt"),
                    "checkpoint_kind": "lora",
                    "metrics": metrics,
                    "extra": {"note": "SYNTHETIC"},
                    "validation_status": "GPU_PENDING",
                    "data_source": "test_fixture",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    beta = sweep_value if sweep_value is not None else 0.1
    status_payload = {
        "run_id": run_id,
        "variant": variant,
        "seed": seed,
        "status": status,
        "task": task,
        "algorithm": "dpo",
        "run_dir": str(run_dir),
        "checkpoint": checkpoint or (str(run_dir / "ckpt") if status != "failed" else None),
        "eval_report": str(report_path) if report_path else None,
        "metrics": {"data_source": "test_fixture"},
        "overrides": {"training.fixed_beta": beta, "seed": seed},
        "error": error,
        "data_source": "test_fixture",
    }
    (run_dir / "status.json").write_text(json.dumps(status_payload), encoding="utf-8")


def test_multi_seed_mean_and_sample_std(tmp_path: Path):
    root = tmp_path / "ablation" / "stat"
    _write_meta(root)
    _write_run(
        root,
        run_id="a-s1",
        variant="fixed_beta_0.1",
        seed=1,
        status="succeeded",
        task="gsm8k",
        metrics={"accuracy": 0.60},
        sweep_value=0.1,
    )
    _write_run(
        root,
        run_id="a-s2",
        variant="fixed_beta_0.1",
        seed=2,
        status="succeeded",
        task="gsm8k",
        metrics={"accuracy": 0.70},
        sweep_value=0.1,
    )
    agg = ResultAggregator(root, baseline_variant="fixed_beta_0.1")
    rows = agg.summary_rows()
    acc = next(r for r in rows if r["metric_name"] == "accuracy")
    assert acc["n_usable"] == 2
    assert abs(acc["mean"] - 0.65) < 1e-12
    expected_std = statistics.stdev([0.60, 0.70])
    assert abs(acc["std"] - expected_std) < 1e-9
    assert abs(acc["std"] - math.sqrt(0.005)) < 1e-9  # ≈ 0.070710678


def test_failures_kept_but_excluded_from_mean(tmp_path: Path):
    root = tmp_path / "ablation" / "fail"
    _write_meta(root)
    _write_run(
        root,
        run_id="ok",
        variant="fixed_beta_0.1",
        seed=1,
        status="succeeded",
        task="gsm8k",
        metrics={"accuracy": 0.8},
        sweep_value=0.1,
    )
    _write_run(
        root,
        run_id="train_fail",
        variant="fixed_beta_0.1",
        seed=2,
        status="failed",
        task="gsm8k",
        metrics=None,
        error="train_boom",
        sweep_value=0.1,
    )
    _write_run(
        root,
        run_id="eval_fail",
        variant="fixed_beta_0.1",
        seed=3,
        status="eval_failed",
        task="gsm8k",
        metrics=None,
        error="eval_boom",
        checkpoint="/tmp/ckpt",
        sweep_value=0.1,
    )
    agg = ResultAggregator(root)
    results = agg.collect()
    statuses = {r["run_id"]: r["status"] for r in results}
    assert statuses["train_fail"] == "failed"
    assert statuses["eval_fail"] == "eval_failed"
    assert statuses["ok"] == "succeeded"
    # 失败行 metric 为空
    fail_rows = [r for r in results if r["run_id"] == "train_fail"]
    assert fail_rows[0]["metric_value"] is None

    summary = next(s for s in agg.summary_rows() if s["metric_name"] == "accuracy")
    assert summary["n_usable"] == 1
    assert summary["mean"] == 0.8
    assert summary["n_failed"] == 2


def test_missing_null_and_single_seed_std(tmp_path: Path):
    root = tmp_path / "ablation" / "miss"
    _write_meta(root)
    _write_run(
        root,
        run_id="dry",
        variant="fixed_beta_0.1",
        seed=1,
        status="dry_run",
        task="gsm8k",
        metrics=None,
        sweep_value=0.1,
    )
    _write_run(
        root,
        run_id="null_metric",
        variant="fixed_beta_0.1",
        seed=2,
        status="succeeded",
        task="gsm8k",
        metrics={"accuracy": None},  # type: ignore[dict-item]
        sweep_value=0.1,
    )
    # 修复：succeeded 但 metrics 含 None → 当作无有效数值
    # 上面 write 会把 None 写进 JSON；aggregator 应忽略
    _write_run(
        root,
        run_id="one",
        variant="fixed_beta_0.1",
        seed=3,
        status="succeeded",
        task="gsm8k",
        metrics={"accuracy": 0.55},
        sweep_value=0.1,
    )
    # 无有效实验的另一 variant
    _write_run(
        root,
        run_id="empty_v",
        variant="fixed_beta_0.5",
        seed=1,
        status="failed",
        task="gsm8k",
        metrics=None,
        sweep_value=0.5,
    )

    agg = ResultAggregator(root)
    summaries = agg.summary_rows()
    # 只有 0.1 / accuracy 有均值
    acc = [s for s in summaries if s["variant"] == "fixed_beta_0.1" and s["metric_name"] == "accuracy"]
    assert len(acc) == 1
    assert acc[0]["n_usable"] == 1
    assert abs(acc[0]["mean"] - 0.55) < 1e-12
    assert acc[0]["std"] is None  # n=1 不伪造
    assert acc[0]["n_missing"] >= 1

    # 失败 variant 不应出现虚假 mean 行（无 metric）
    fake = [s for s in summaries if s["variant"] == "fixed_beta_0.5" and s.get("mean") is not None]
    assert fake == []


def test_task_metric_isolation_and_plots(tmp_path: Path):
    root = tmp_path / "ablation" / "tasks"
    _write_meta(root, sweep={"training.fixed_beta": [0.1]}, sweep_parameters=["training.fixed_beta"])
    _write_run(
        root,
        run_id="g1",
        variant="fixed_beta_0.1",
        seed=1,
        status="succeeded",
        task="gsm8k",
        metrics={"accuracy": 0.9},
        sweep_value=0.1,
    )
    _write_run(
        root,
        run_id="h1",
        variant="fixed_beta_0.1",
        seed=1,
        status="succeeded",
        task="humaneval",
        metrics={"pass_at_1": 0.1},
        sweep_value=0.1,
    )
    agg = ResultAggregator(root, require_plots=True)
    paths = agg.write_all()
    summaries = agg.summary_rows()
    assert {s["task"] for s in summaries} == {"gsm8k", "humaneval"}
    assert {s["metric_name"] for s in summaries} == {"accuracy", "pass_at_1"}

    plots = list(Path(paths["plots_dir"]).glob("*.png"))
    names = {p.name for p in plots}
    assert "gsm8k_accuracy.png" in names
    assert "humaneval_pass_at_1.png" in names
    for p in plots:
        assert p.stat().st_size > 0

    md = Path(paths["report_md"]).read_text(encoding="utf-8")
    assert "gsm8k_accuracy.png" in md
    assert "humaneval_pass_at_1.png" in md
    assert "data_source=test_fixture" in md


def test_report_traceable_and_recomputable(tmp_path: Path):
    root = tmp_path / "ablation" / "trace"
    _write_meta(root)
    _write_run(
        root,
        run_id="r1",
        variant="fixed_beta_0.1",
        seed=1,
        status="succeeded",
        task="gsm8k",
        metrics={"accuracy": 0.2},
        sweep_value=0.1,
    )
    _write_run(
        root,
        run_id="r2",
        variant="fixed_beta_0.1",
        seed=2,
        status="succeeded",
        task="gsm8k",
        metrics={"accuracy": 0.4},
        sweep_value=0.1,
    )
    agg = ResultAggregator(root, baseline_variant="fixed_beta_0.1")
    paths = agg.write_all()
    # 从 results.csv 重算
    with open(paths["results_csv"], encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    vals = [
        float(r["metric_value"])
        for r in rows
        if r["status"] == "succeeded" and r["metric_name"] == "accuracy"
    ]
    assert abs(statistics.fmean(vals) - 0.3) < 1e-12
    with open(paths["summary_csv"], encoding="utf-8") as f:
        summary = list(csv.DictReader(f))
    s = next(x for x in summary if x["metric_name"] == "accuracy")
    assert abs(float(s["mean"]) - 0.3) < 1e-12
    assert "r1" in Path(paths["report_md"]).read_text(encoding="utf-8")


def test_idempotent_regenerate(tmp_path: Path):
    root = tmp_path / "ablation" / "idem"
    _write_meta(root)
    _write_run(
        root,
        run_id="r1",
        variant="fixed_beta_0.1",
        seed=1,
        status="succeeded",
        task="gsm8k",
        metrics={"accuracy": 0.5},
        sweep_value=0.1,
    )
    status_before = (root / "runs" / "r1" / "status.json").read_text(encoding="utf-8")
    a1 = ResultAggregator(root).write_all()
    s1 = Path(a1["summary_csv"]).read_text(encoding="utf-8")
    a2 = ResultAggregator(root).write_all()
    s2 = Path(a2["summary_csv"]).read_text(encoding="utf-8")
    assert s1 == s2
    assert (root / "runs" / "r1" / "status.json").read_text(encoding="utf-8") == status_before


def test_matplotlib_missing_raises_when_required(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "ablation" / "nopy"
    _write_meta(root)
    _write_run(
        root,
        run_id="r1",
        variant="fixed_beta_0.1",
        seed=1,
        status="succeeded",
        task="gsm8k",
        metrics={"accuracy": 0.5},
        sweep_value=0.1,
    )

    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(RuntimeError, match="matplotlib"):
        ResultAggregator(root, require_plots=True).write_plots()
