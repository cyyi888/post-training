"""【设备：CPU】消融 Planner / Runner Mock 集成测试。

不启动真实训练。验证：3β×3seed=9、dry-run、失败重试、评测失败、
中断恢复、已成功跳过。生产 Runner 仍保留 subprocess → scripts/train.py。
"""

from __future__ import annotations

import json
from pathlib import Path

from omegaconf import OmegaConf

from src.experiment.planner import ExperimentPlanner, stable_run_id
from src.experiment.runner import EvalResultBox, ExperimentRunner, TrainResultBox
from src.experiment.types import RunStatus


def _ablation_cfg(tmp_path: Path, **exp_extra):
    exp = {
        "name": "ablation_dpo_beta",
        "dry_run": False,
        "max_retries": 1,
        "skip_completed": True,
        "auto_eval": True,
        "eval_task": "gsm8k",
        "retry_backoff_sec": 0,
        "seeds": [12, 34, 56],
        "sweep": {"training.fixed_beta": [0.1, 0.5, 1.0]},
        "baseline_overrides": {
            "training.loss_type": "dpo",
            "training.beta_mode": "fixed",
        },
    }
    exp.update(exp_extra)
    return OmegaConf.create(
        {
            "output_dir": str(tmp_path),
            "seed": 0,
            "model": {"name": "Qwen/Qwen2.5-0.5B-Instruct"},
            "data": {"name": "dpo_demo"},
            "training": {"algorithm": "dpo", "fixed_beta": 0.1},
            "experiment": exp,
        }
    )


def test_planner_expands_3_beta_x_3_seed(tmp_path: Path):
    cfg = _ablation_cfg(tmp_path)
    runs = ExperimentPlanner(cfg).plan()
    assert len(runs) == 9
    betas = sorted({r.overrides["training.fixed_beta"] for r in runs})
    seeds = sorted({r.seed for r in runs})
    assert betas == [0.1, 0.5, 1.0]
    assert seeds == [12, 34, 56]
    # 稳定 run_id
    ids = [r.run_id for r in runs]
    assert len(ids) == len(set(ids))
    again = ExperimentPlanner(cfg).plan()
    assert [r.run_id for r in again] == ids


def test_stable_run_id_deterministic():
    a = stable_run_id("exp", "v", 12, {"training.fixed_beta": 0.1, "seed": 12})
    b = stable_run_id("exp", "v", 12, {"seed": 12, "training.fixed_beta": 0.1})
    assert a == b


def test_dry_run_does_not_call_train(tmp_path: Path):
    cfg = _ablation_cfg(tmp_path, dry_run=True)
    called = {"n": 0}

    def train_fn(run):
        called["n"] += 1
        return TrainResultBox(0, checkpoint="x")

    records = ExperimentRunner(cfg, train_fn=train_fn).run_all()
    assert called["n"] == 0
    assert len(records) == 9
    assert all(r.status == RunStatus.DRY_RUN.value for r in records)
    assert (tmp_path / "ablation" / "ablation_dpo_beta" / "plan.json").is_file()


def test_train_success_and_eval(tmp_path: Path):
    cfg = _ablation_cfg(tmp_path, dry_run=False, seeds=[12], sweep={"training.fixed_beta": [0.1]})

    def train_fn(run):
        ckpt = Path(run.run_dir) / "outputs" / "checkpoints" / "dpo"
        ckpt.mkdir(parents=True)
        (ckpt / "adapter_config.json").write_text("{}", encoding="utf-8")
        return TrainResultBox(0, checkpoint=str(ckpt), metrics={"train_loss": 0.1})

    def eval_fn(run, checkpoint):
        report = Path(run.run_dir) / "eval_report.json"
        payload = {
            "task": "gsm8k",
            "checkpoint": checkpoint,
            "metrics": {"accuracy": 0.4},
            "data_source": "test_fixture",
        }
        report.write_text(json.dumps(payload), encoding="utf-8")
        return EvalResultBox(True, report_path=str(report), metrics={"accuracy": 0.4})

    recs = ExperimentRunner(cfg, train_fn=train_fn, eval_fn=eval_fn).run_all()
    assert len(recs) == 1
    assert recs[0].status == RunStatus.SUCCEEDED.value
    assert recs[0].checkpoint
    assert recs[0].metrics.get("accuracy") == 0.4


def test_train_failure_retries(tmp_path: Path):
    cfg = _ablation_cfg(
        tmp_path, dry_run=False, seeds=[12], sweep={"training.fixed_beta": [0.1]}, max_retries=2
    )
    calls = {"n": 0}

    def train_fn(run):
        calls["n"] += 1
        return TrainResultBox(1, error="boom")

    recs = ExperimentRunner(cfg, train_fn=train_fn, eval_fn=lambda *a: EvalResultBox(True)).run_all()
    assert recs[0].status == RunStatus.FAILED.value
    # 首次 + 2 次重试 = 3
    assert calls["n"] == 3
    assert recs[0].attempts == 3


def test_train_ok_eval_fail(tmp_path: Path):
    cfg = _ablation_cfg(tmp_path, dry_run=False, seeds=[12], sweep={"training.fixed_beta": [0.1]})

    def train_fn(run):
        return TrainResultBox(0, checkpoint=str(tmp_path / "ckpt"))

    def eval_fn(run, checkpoint):
        return EvalResultBox(False, error="eval_boom")

    recs = ExperimentRunner(cfg, train_fn=train_fn, eval_fn=eval_fn).run_all()
    assert recs[0].status == RunStatus.EVAL_FAILED.value


def test_skip_completed_on_reschedule(tmp_path: Path):
    cfg = _ablation_cfg(tmp_path, dry_run=False, seeds=[12], sweep={"training.fixed_beta": [0.1]})
    n_train = {"n": 0}

    def train_fn(run):
        n_train["n"] += 1
        ckpt = Path(run.run_dir) / "ckpt"
        ckpt.mkdir(parents=True, exist_ok=True)
        (ckpt / "adapter_config.json").write_text("{}", encoding="utf-8")
        return TrainResultBox(0, checkpoint=str(ckpt))

    def eval_fn(run, checkpoint):
        return EvalResultBox(True, metrics={"accuracy": 0.3}, report_path=str(Path(run.run_dir) / "e.json"))

    runner = ExperimentRunner(cfg, train_fn=train_fn, eval_fn=eval_fn)
    first = runner.run_all()
    assert first[0].status == RunStatus.SUCCEEDED.value
    assert n_train["n"] == 1

    second = ExperimentRunner(cfg, train_fn=train_fn, eval_fn=eval_fn).run_all()
    assert second[0].status == RunStatus.SKIPPED.value
    assert n_train["n"] == 1  # 未再次训练


def test_interrupt_resume_failed_then_ok(tmp_path: Path):
    """模拟中断：第一次调度全部失败；切换 mode 后再次调度可重跑。"""
    cfg = _ablation_cfg(tmp_path, dry_run=False, seeds=[12], sweep={"training.fixed_beta": [0.1]})
    state = {"mode": "fail"}

    def train_fn(run):
        if state["mode"] == "fail":
            return TrainResultBox(1, error="interrupted")
        ckpt = Path(run.run_dir) / "ckpt"
        ckpt.mkdir(parents=True, exist_ok=True)
        (ckpt / "adapter_config.json").write_text("{}", encoding="utf-8")
        return TrainResultBox(0, checkpoint=str(ckpt))

    def eval_fn(run, checkpoint):
        return EvalResultBox(True, metrics={"accuracy": 0.5})

    r1 = ExperimentRunner(cfg, train_fn=train_fn, eval_fn=eval_fn).run_all()
    assert r1[0].status == RunStatus.FAILED.value

    state["mode"] = "ok"
    r2 = ExperimentRunner(cfg, train_fn=train_fn, eval_fn=eval_fn).run_all()
    assert r2[0].status == RunStatus.SUCCEEDED.value


def test_production_runner_has_subprocess_train_method():
    """确保生产路径方法存在（不被 Mock 专用实现替换）。"""
    assert hasattr(ExperimentRunner, "_subprocess_train")
    assert hasattr(ExperimentRunner, "_subprocess_eval")
