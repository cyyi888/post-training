"""ExperimentRunner：独立进程调用训练入口、状态管理、重试与自动评测。"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from src.experiment.planner import ExperimentPlanner, _exp_section
from src.experiment.types import PlannedRun, RunRecord, RunStatus

logger = logging.getLogger("posttrainlab.experiment.runner")

TrainCallable = Callable[[PlannedRun], Any]
EvalCallable = Callable[[PlannedRun, str], Any]


class TrainResultBox:
    def __init__(
        self,
        exit_code: int,
        checkpoint: str | None = None,
        error: str | None = None,
        metrics: dict[str, Any] | None = None,
    ):
        self.exit_code = int(exit_code)
        self.checkpoint = checkpoint
        self.error = error
        self.metrics = metrics or {}


class EvalResultBox:
    def __init__(
        self,
        ok: bool,
        report_path: str | None = None,
        metrics: dict[str, Any] | None = None,
        error: str | None = None,
    ):
        self.ok = bool(ok)
        self.report_path = report_path
        self.metrics = metrics or {}
        self.error = error


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


class ExperimentRunner:
    """
    执行 Planner 产出的 runs。

    - 生产：``subprocess`` 调用 ``scripts/train.py`` + Hydra overrides
    - 测试：注入 ``train_fn`` / ``eval_fn``（Mock Runner）
    - ``dry_run``：只写计划与 dry_run 状态，不启动训练
    """

    def __init__(
        self,
        cfg: Any,
        *,
        train_fn: TrainCallable | None = None,
        eval_fn: EvalCallable | None = None,
        python_executable: str | None = None,
    ):
        self.cfg = cfg
        self.exp = _exp_section(cfg)
        self.planner = ExperimentPlanner(cfg)
        self.train_fn = train_fn
        self.eval_fn = eval_fn
        self.python = python_executable or sys.executable
        self.records: list[RunRecord] = []

    @property
    def dry_run(self) -> bool:
        # 默认 True：本地开发防误训；GPU 上显式 experiment.dry_run=false
        val = self.exp.get("dry_run", True)
        return bool(val)

    @property
    def max_retries(self) -> int:
        return max(0, int(self.exp.get("max_retries", 1)))

    @property
    def skip_completed(self) -> bool:
        return bool(self.exp.get("skip_completed", True))

    @property
    def auto_eval(self) -> bool:
        return bool(self.exp.get("auto_eval", True))

    def run_all(self, runs: list[PlannedRun] | None = None) -> list[RunRecord]:
        planned = runs if runs is not None else self.planner.plan()
        self.planner.write_snapshots(planned)
        self.records = []
        for run in planned:
            rec = self.run_one(run)
            self.records.append(rec)
        self._write_index()
        return self.records

    def _algorithm(self) -> str | None:
        t = self.cfg.get("training") if hasattr(self.cfg, "get") else None
        if t is None:
            return None
        if hasattr(t, "get"):
            return t.get("algorithm")
        return getattr(t, "algorithm", None)

    def _base_record(self, run: PlannedRun, *, status: str, attempts: int = 0) -> RunRecord:
        return RunRecord(
            run_id=run.run_id,
            variant=run.variant,
            seed=run.seed,
            status=status,
            run_dir=str(run.run_dir),
            hydra_overrides=list(run.hydra_overrides),
            overrides=dict(run.overrides),
            algorithm=str(self._algorithm() or "") or None,
            task=str(self.exp.get("eval_task") or "") or None,
            attempts=attempts,
        )

    def run_one(self, run: PlannedRun) -> RunRecord:
        run_dir = Path(run.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        status_path = run_dir / "status.json"
        eval_only = bool(self.exp.get("eval_only", False))

        if self.skip_completed and status_path.is_file() and not eval_only:
            prev = RunRecord.load(status_path)
            if prev.status in (RunStatus.SUCCEEDED.value, RunStatus.SKIPPED.value):
                logger.info("Skip completed run_id=%s", run.run_id)
                prev.status = RunStatus.SKIPPED.value
                prev.error = "skipped_already_succeeded"
                # 补齐追溯字段
                if not prev.overrides:
                    prev.overrides = dict(run.overrides)
                if not prev.algorithm:
                    prev.algorithm = str(self._algorithm() or "") or None
                prev.save(status_path)
                return prev

        if self.dry_run and not eval_only:
            rec = self._base_record(run, status=RunStatus.DRY_RUN.value)
            rec.metrics = {"status": "dry_run"}
            rec.data_source = rec.metrics.get("data_source")
            rec.save(status_path)
            logger.info("Dry-run run_id=%s overrides=%s", run.run_id, run.hydra_overrides)
            return rec

        # 仅重试评测：保留既有 checkpoint，不重新训练
        if eval_only and status_path.is_file():
            prev = RunRecord.load(status_path)
            if prev.status in (
                RunStatus.EVAL_FAILED.value,
                RunStatus.TRAIN_OK.value,
            ) and prev.checkpoint:
                return self._run_eval_only(run, prev)

        attempts = 0
        last_error: str | None = None
        while attempts <= self.max_retries:
            attempts += 1
            rec = self._base_record(run, status=RunStatus.RUNNING.value, attempts=attempts)
            rec.save(status_path)
            try:
                train_out = self._execute_train(run)
            except Exception as exc:
                last_error = str(exc)
                train_out = TrainResultBox(exit_code=1, error=last_error)

            rec.train_exit_code = train_out.exit_code
            rec.checkpoint = train_out.checkpoint
            rec.metrics.update(train_out.metrics)
            if train_out.metrics.get("data_source"):
                rec.data_source = str(train_out.metrics["data_source"])

            if train_out.exit_code != 0:
                rec.status = RunStatus.FAILED.value
                rec.error = train_out.error or f"train_exit={train_out.exit_code}"
                last_error = rec.error
                rec.save(status_path)
                logger.warning(
                    "Train failed run_id=%s attempt=%s err=%s",
                    run.run_id,
                    attempts,
                    rec.error,
                )
                if attempts <= self.max_retries:
                    time.sleep(float(self.exp.get("retry_backoff_sec", 0)))
                    continue
                return rec

            rec.status = RunStatus.TRAIN_OK.value
            rec.save(status_path)

            if not self.auto_eval:
                # 无评测时不算任务成功指标源；状态记 train_ok
                return rec

            ckpt = train_out.checkpoint
            if not ckpt:
                rec.status = RunStatus.EVAL_FAILED.value
                rec.error = "train_ok_but_missing_checkpoint"
                rec.save(status_path)
                return rec

            rec = self._apply_eval(run, rec, ckpt)
            if (
                rec.status == RunStatus.EVAL_FAILED.value
                and bool(self.exp.get("retry_on_eval_failure", False))
                and attempts <= self.max_retries
            ):
                continue
            return rec

        rec = self._base_record(run, status=RunStatus.FAILED.value, attempts=attempts)
        rec.error = last_error or "exhausted_retries"
        rec.save(status_path)
        return rec

    def _run_eval_only(self, run: PlannedRun, prev: RunRecord) -> RunRecord:
        """训练已成功、评测失败时仅重试评测，不抹掉 checkpoint。"""
        ckpt = prev.checkpoint
        assert ckpt
        rec = prev
        rec.overrides = dict(run.overrides) or dict(prev.overrides)
        rec.algorithm = rec.algorithm or str(self._algorithm() or "") or None
        rec.task = str(self.exp.get("eval_task") or rec.task or "") or None
        rec.status = RunStatus.TRAIN_OK.value
        rec.error = None
        rec.save()
        return self._apply_eval(run, rec, ckpt)

    def _apply_eval(self, run: PlannedRun, rec: RunRecord, checkpoint: str) -> RunRecord:
        try:
            eval_out = self._execute_eval(run, checkpoint)
        except Exception as exc:
            eval_out = EvalResultBox(ok=False, error=str(exc))

        if not eval_out.ok:
            rec.status = RunStatus.EVAL_FAILED.value
            rec.error = eval_out.error or "eval_failed"
            rec.eval_report = eval_out.report_path
            rec.metrics.update(eval_out.metrics)
            if eval_out.metrics.get("data_source"):
                rec.data_source = str(eval_out.metrics["data_source"])
            rec.checkpoint = checkpoint
            rec.save()
            return rec

        rec.status = RunStatus.SUCCEEDED.value
        rec.eval_report = eval_out.report_path
        rec.metrics.update(eval_out.metrics)
        rec.checkpoint = checkpoint
        rec.error = None
        if eval_out.metrics.get("data_source"):
            rec.data_source = str(eval_out.metrics["data_source"])
        # 从 EvaluationResult 风格报告补 task
        if rec.eval_report and Path(rec.eval_report).is_file():
            try:
                payload = json.loads(Path(rec.eval_report).read_text(encoding="utf-8"))
                if payload.get("task"):
                    rec.task = str(payload["task"])
                if payload.get("data_source"):
                    rec.data_source = str(payload["data_source"])
            except Exception:
                pass
        rec.save()
        logger.info("Succeeded run_id=%s ckpt=%s", run.run_id, checkpoint)
        return rec

    def _execute_train(self, run: PlannedRun) -> TrainResultBox:
        if self.train_fn is not None:
            out = self.train_fn(run)
            if isinstance(out, TrainResultBox):
                return out
            # 允许返回 dict
            return TrainResultBox(
                exit_code=int(out.get("exit_code", 0)),
                checkpoint=out.get("checkpoint"),
                error=out.get("error"),
                metrics=dict(out.get("metrics") or {}),
            )
        return self._subprocess_train(run)

    def _execute_eval(self, run: PlannedRun, checkpoint: str) -> EvalResultBox:
        if self.eval_fn is not None:
            out = self.eval_fn(run, checkpoint)
            if isinstance(out, EvalResultBox):
                return out
            return EvalResultBox(
                ok=bool(out.get("ok", False)),
                report_path=out.get("report_path"),
                metrics=dict(out.get("metrics") or {}),
                error=out.get("error"),
            )
        return self._subprocess_eval(run, checkpoint)

    def _subprocess_train(self, run: PlannedRun) -> TrainResultBox:
        """生产路径：调用现有 ``scripts/train.py``。"""
        script = _repo_root() / "scripts" / "train.py"
        cmd = [self.python, str(script), *run.hydra_overrides]
        # 追加算法/数据等来自父配置的关键 override（若 planner 未展开）
        log_path = Path(run.run_dir) / "train.log"
        logger.info("Launch train: %s", " ".join(cmd))
        env = os.environ.copy()
        env.setdefault("WANDB_DISABLED", "true")
        with log_path.open("w", encoding="utf-8") as logf:
            proc = subprocess.run(
                cmd,
                cwd=str(_repo_root()),
                env=env,
                stdout=logf,
                stderr=subprocess.STDOUT,
                check=False,
            )
        ckpt = self._discover_checkpoint(Path(run.run_dir))
        if proc.returncode != 0:
            return TrainResultBox(
                exit_code=proc.returncode,
                checkpoint=ckpt,
                error=f"subprocess_exit={proc.returncode}",
            )
        if not ckpt:
            return TrainResultBox(
                exit_code=1,
                error="train_finished_but_checkpoint_missing",
            )
        return TrainResultBox(exit_code=0, checkpoint=ckpt)

    def _subprocess_eval(self, run: PlannedRun, checkpoint: str) -> EvalResultBox:
        """生产路径：调用 ``scripts/eval.py`` 指向该 checkpoint。"""
        script = _repo_root() / "scripts" / "eval.py"
        report_dir = Path(run.run_dir) / "eval"
        report_dir.mkdir(parents=True, exist_ok=True)
        task = str(self.exp.get("eval_task") or "gsm8k")
        overrides = [
            f"eval={task}" if task in ("gsm8k", "humaneval", "default") else "eval=default",
            f"eval.task={task}",
            f"eval.checkpoint={checkpoint}",
            f"eval.report_name=eval_report.json",
            f"output_dir={run.run_dir}/outputs",
        ]
        cmd = [self.python, str(script), *overrides]
        log_path = Path(run.run_dir) / "eval.log"
        logger.info("Launch eval: %s", " ".join(cmd))
        env = os.environ.copy()
        env.setdefault("WANDB_DISABLED", "true")
        with log_path.open("w", encoding="utf-8") as logf:
            proc = subprocess.run(
                cmd,
                cwd=str(_repo_root()),
                env=env,
                stdout=logf,
                stderr=subprocess.STDOUT,
                check=False,
            )
        # Hydra 会把报告写到自己的 output；同时尝试 run_dir/eval
        report = self._find_eval_report(Path(run.run_dir))
        if proc.returncode != 0:
            return EvalResultBox(
                ok=False,
                report_path=str(report) if report else None,
                error=f"eval_exit={proc.returncode}",
            )
        metrics: dict[str, Any] = {}
        if report and report.is_file():
            try:
                payload = json.loads(report.read_text(encoding="utf-8"))
                metrics = dict(payload.get("metrics") or {})
                # 兼容扁平报告
                for k, v in payload.items():
                    if k not in ("task", "checkpoint", "checkpoint_kind", "metrics", "extra", "report_path") and isinstance(
                        v, (int, float)
                    ):
                        metrics.setdefault(k, v)
            except Exception:
                pass
        return EvalResultBox(ok=True, report_path=str(report) if report else None, metrics=metrics)

    @staticmethod
    def _discover_checkpoint(run_dir: Path) -> str | None:
        from src.core.base import BaseTrainer

        outputs = run_dir / "outputs"
        if not outputs.is_dir():
            outputs = run_dir
        # 常见：outputs/checkpoints/<stage>/...
        candidates: list[str] = []
        for p in outputs.rglob("adapter_config.json"):
            candidates.append(str(p.parent))
        for p in outputs.rglob("config.json"):
            # 排除 tokenizer 等；需要像 HF 模型目录
            parent = p.parent
            if (parent / "model.safetensors").exists() or (parent / "pytorch_model.bin").exists() or list(
                parent.glob("checkpoint-*")
            ):
                candidates.append(str(parent))
        # checkpoint-* 目录
        for p in outputs.rglob("checkpoint-*"):
            if p.is_dir():
                found = BaseTrainer.latest_checkpoint(str(p.parent))
                if found:
                    candidates.append(found)
        if not candidates:
            return None
        # 取最近修改
        return str(max(map(Path, candidates), key=lambda x: x.stat().st_mtime))

    @staticmethod
    def _find_eval_report(run_dir: Path) -> Path | None:
        hits = list(run_dir.rglob("eval_report.json"))
        if not hits:
            return None
        return max(hits, key=lambda p: p.stat().st_mtime)

    def _write_index(self) -> Path:
        root = self.planner.root
        root.mkdir(parents=True, exist_ok=True)
        path = root / "runs_index.json"
        path.write_text(
            json.dumps([r.to_dict() for r in self.records], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path
