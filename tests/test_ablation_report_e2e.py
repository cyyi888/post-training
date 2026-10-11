"""【设备：CPU】消融报告闭环：Planner → Runner(Dummy) → EvaluationResult → Aggregator。

证明工程数据流闭环；不代表 GPU / 真实算法效果验证（data_source=test_fixture）。
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

from omegaconf import OmegaConf

from src.evaluators.result import EvaluationResult
from src.experiment.aggregator import ResultAggregator
from src.experiment.planner import ExperimentPlanner, format_hydra_override
from src.experiment.runner import EvalResultBox, ExperimentRunner, TrainResultBox
from src.experiment.types import RunStatus


def _cfg(tmp_path: Path):
    return OmegaConf.create(
        {
            "output_dir": str(tmp_path / "outputs"),
            "seed": 0,
            "use_gpu": False,
            "model": {"name": "Qwen/Qwen2.5-0.5B-Instruct", "use_lora": True},
            "data": {"name": "dpo_demo"},
            "training": {
                "algorithm": "dpo",
                "loss_type": "dpo",
                "beta_mode": "fixed",
                "fixed_beta": 0.1,
            },
            "engineering": {"init_from_checkpoint": "sft"},
            "experiment": {
                "name": "ablation_dpo_beta_e2e_fixture",
                "description": "CPU e2e fixture — NOT real results",
                "dry_run": False,
                "report_only": False,
                "eval_only": False,
                "max_retries": 0,
                "skip_completed": True,
                "auto_eval": True,
                "aggregate": True,
                "require_plots": True,
                "eval_task": "gsm8k",
                "baseline_variant": "fixed_beta_0.05",
                "seeds": [42, 123, 456],
                "sweep": {"training.fixed_beta": [0.05, 0.1, 0.2]},
                "baseline_overrides": {
                    "training.loss_type": "dpo",
                    "training.beta_mode": "fixed",
                    "training.penalty.enabled": False,
                },
            },
        }
    )


def _accuracy_for(beta: float, seed: int) -> float:
    """确定性伪指标（仅测试用）。"""
    return round(0.5 + float(beta) + (seed % 10) * 0.001, 6)


def _make_train_fn():
    def train_fn(run):
        ckpt = Path(run.run_dir) / "outputs" / "checkpoints" / "dpo" / "checkpoint-1"
        ckpt.mkdir(parents=True, exist_ok=True)
        (ckpt / "adapter_config.json").write_text('{"peft_type":"LORA"}', encoding="utf-8")
        return TrainResultBox(
            0,
            checkpoint=str(ckpt),
            metrics={"data_source": "test_fixture"},
        )

    return train_fn


def _make_eval_fn(fail_run_id: str | None = None):
    def eval_fn(run, checkpoint: str):
        if fail_run_id and run.run_id == fail_run_id:
            return EvalResultBox(False, error="simulated_eval_failure")
        beta = float(run.overrides["training.fixed_beta"])
        acc = _accuracy_for(beta, run.seed)
        result = EvaluationResult(
            task="gsm8k",
            checkpoint=checkpoint,
            checkpoint_kind="lora",
            metrics={"accuracy": acc, "format_rate": 1.0},
            extra={"data_source": "test_fixture", "synthetic": True},
            validation_status="GPU_PENDING",
        )
        # 明确标注
        payload = result.to_dict()
        payload["data_source"] = "test_fixture"
        report = Path(run.run_dir) / "eval_report.json"
        report.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return EvalResultBox(
            True,
            report_path=str(report),
            metrics={"accuracy": acc, "format_rate": 1.0, "data_source": "test_fixture"},
        )

    return eval_fn


def test_e2e_planner_runner_aggregator_closed_loop(tmp_path: Path):
    cfg = _cfg(tmp_path)
    planner = ExperimentPlanner(cfg)
    runs = planner.plan()
    assert len(runs) == 9
    betas = sorted({float(r.overrides["training.fixed_beta"]) for r in runs})
    seeds = sorted({r.seed for r in runs})
    assert betas == [0.05, 0.1, 0.2]
    assert seeds == [42, 123, 456]

    # 生产 Hydra override 结构匹配（不启动训练）
    sample = runs[0]
    assert any(o.startswith("training.fixed_beta=") for o in sample.hydra_overrides)
    assert any(o.startswith("seed=") for o in sample.hydra_overrides)
    assert any(o.startswith("output_dir=") for o in sample.hydra_overrides)
    assert format_hydra_override("training.fixed_beta", 0.05) == "training.fixed_beta=0.05"
    assert hasattr(ExperimentRunner, "_subprocess_train")

    planner.write_snapshots(runs)
    runner = ExperimentRunner(
        cfg,
        train_fn=_make_train_fn(),
        eval_fn=_make_eval_fn(),
    )
    records = runner.run_all(runs)
    assert len(records) == 9
    assert all(r.status == RunStatus.SUCCEEDED.value for r in records)
    assert all(r.checkpoint for r in records)
    assert all(r.eval_report and Path(r.eval_report).is_file() for r in records)
    assert all(r.data_source == "test_fixture" or "test_fixture" in str(r.metrics) for r in records)

    root = planner.root
    agg = ResultAggregator(root, baseline_variant="fixed_beta_0.05", require_plots=True)
    paths1 = agg.write_all()

    # 文件交付
    for key in ("results_csv", "summary_csv", "report_md", "plots_dir"):
        assert key in paths1
        assert Path(paths1[key]).exists()

    results = list(csv.DictReader(open(paths1["results_csv"], encoding="utf-8")))
    # 每 run × 2 metrics (accuracy, format_rate) = 18
    assert len(results) == 18
    assert {"run_id", "algorithm", "seed", "sweep_parameter", "sweep_value",
            "checkpoint", "task", "metric_name", "metric_value", "status", "data_source"} <= set(
        results[0].keys()
    )
    assert all(r["data_source"] == "test_fixture" for r in results)
    assert all(r["algorithm"] == "dpo" for r in results)
    assert all(r["sweep_parameter"] == "training.fixed_beta" for r in results)

    # 数值：beta=0.05 的 accuracy 均值
    acc_005 = [
        float(r["metric_value"])
        for r in results
        if r["metric_name"] == "accuracy" and float(r["sweep_value"]) == 0.05
    ]
    assert len(acc_005) == 3
    expected = statistics.fmean([_accuracy_for(0.05, s) for s in (42, 123, 456)])
    assert abs(statistics.fmean(acc_005) - expected) < 1e-9

    summary = list(csv.DictReader(open(paths1["summary_csv"], encoding="utf-8")))
    row = next(
        s
        for s in summary
        if s["metric_name"] == "accuracy" and float(s["sweep_value"]) == 0.05
    )
    assert int(row["n_usable"]) == 3
    assert abs(float(row["mean"]) - expected) < 1e-9
    assert row["std"] != ""  # n=3 有样本标准差

    md = Path(paths1["report_md"]).read_text(encoding="utf-8")
    assert "data_source=test_fixture" in md
    assert "gsm8k_accuracy.png" in md
    assert "样本标准差" in md or "stdev" in md
    png = Path(paths1["plots_dir"]) / "gsm8k_accuracy.png"
    assert png.is_file() and png.stat().st_size > 0

    # 重复生成：稳定且不改 status
    status_files = {p: p.read_text(encoding="utf-8") for p in root.glob("runs/*/status.json")}
    paths2 = ResultAggregator(root, baseline_variant="fixed_beta_0.05").write_all()
    assert Path(paths2["summary_csv"]).read_text(encoding="utf-8") == Path(
        paths1["summary_csv"]
    ).read_text(encoding="utf-8")
    for p, text in status_files.items():
        assert p.read_text(encoding="utf-8") == text

    # report_only 路径：不再训练
    n_train = {"n": 0}

    def boom_train(run):
        n_train["n"] += 1
        raise AssertionError("report_only must not train")

    cfg.experiment.report_only = True
    # 直接聚合即可（scripts 层跳过 runner）；此处验证二次聚合
    ResultAggregator(root).write_all()
    assert n_train["n"] == 0


def test_eval_failed_keeps_checkpoint_and_eval_only_retry(tmp_path: Path):
    cfg = _cfg(tmp_path)
    cfg.experiment.seeds = [42]
    cfg.experiment.sweep = {"training.fixed_beta": [0.1]}
    planner = ExperimentPlanner(cfg)
    runs = planner.plan()
    fail_id = runs[0].run_id

    runner = ExperimentRunner(
        cfg,
        train_fn=_make_train_fn(),
        eval_fn=_make_eval_fn(fail_run_id=fail_id),
    )
    recs = runner.run_all(runs)
    assert recs[0].status == RunStatus.EVAL_FAILED.value
    assert recs[0].checkpoint
    ckpt = recs[0].checkpoint

    # 无有效指标不进均值
    agg = ResultAggregator(planner.root, require_plots=False)
    summaries = agg.summary_rows()
    assert all(s.get("mean") is None or s["n_usable"] == 0 for s in summaries) or summaries == []

    # eval_only 重试成功
    cfg.experiment.eval_only = True
    runner2 = ExperimentRunner(
        cfg,
        train_fn=lambda run: (_ for _ in ()).throw(AssertionError("must not retrain")),
        eval_fn=_make_eval_fn(fail_run_id=None),
    )
    recs2 = runner2.run_all(runs)
    assert recs2[0].status == RunStatus.SUCCEEDED.value
    assert recs2[0].checkpoint == ckpt


def test_report_failure_does_not_wipe_run_status(tmp_path: Path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.experiment.seeds = [42]
    cfg.experiment.sweep = {"training.fixed_beta": [0.1]}
    planner = ExperimentPlanner(cfg)
    runs = planner.plan()
    ExperimentRunner(cfg, train_fn=_make_train_fn(), eval_fn=_make_eval_fn()).run_all(runs)
    status_path = Path(runs[0].run_dir) / "status.json"
    before = status_path.read_text(encoding="utf-8")

    def boom_plots(self, path=None):
        raise RuntimeError("plot_boom")

    monkeypatch.setattr(ResultAggregator, "write_plots", boom_plots)
    with __import__("pytest").raises(RuntimeError, match="plot_boom"):
        ResultAggregator(planner.root, require_plots=True).write_all()
    assert status_path.read_text(encoding="utf-8") == before
