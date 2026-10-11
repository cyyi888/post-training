#!/usr/bin/env python
"""生成 CPU 消融报告样例产物（data_source=test_fixture，非真实结论）。"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluators.result import EvaluationResult
from src.experiment.aggregator import ResultAggregator
from src.experiment.planner import ExperimentPlanner
from src.experiment.runner import EvalResultBox, ExperimentRunner, TrainResultBox


def main() -> None:
    out_base = ROOT / "outputs" / "ablation_demo_cpu_fixture"
    if out_base.exists():
        shutil.rmtree(out_base)

    cfg = OmegaConf.create(
        {
            "output_dir": str(out_base),
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
                "name": "ablation_dpo_beta_cpu_fixture",
                "description": "CPU demo fixture — NOT real algorithm results",
                "dry_run": False,
                "max_retries": 0,
                "skip_completed": True,
                "auto_eval": True,
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

    def acc(beta: float, seed: int) -> float:
        return round(0.5 + float(beta) + (seed % 10) * 0.001, 6)

    def train_fn(run):
        ckpt = Path(run.run_dir) / "outputs" / "checkpoints" / "dpo" / "checkpoint-1"
        ckpt.mkdir(parents=True, exist_ok=True)
        (ckpt / "adapter_config.json").write_text('{"peft_type":"LORA"}', encoding="utf-8")
        return TrainResultBox(0, checkpoint=str(ckpt), metrics={"data_source": "test_fixture"})

    def eval_fn(run, checkpoint: str):
        beta = float(run.overrides["training.fixed_beta"])
        a = acc(beta, run.seed)
        result = EvaluationResult(
            task="gsm8k",
            checkpoint=checkpoint,
            checkpoint_kind="lora",
            metrics={"accuracy": a, "format_rate": 1.0},
            extra={"synthetic": True},
            validation_status="GPU_PENDING",
        )
        payload = result.to_dict()
        payload["data_source"] = "test_fixture"
        report = Path(run.run_dir) / "eval_report.json"
        report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return EvalResultBox(
            True,
            report_path=str(report),
            metrics={"accuracy": a, "format_rate": 1.0, "data_source": "test_fixture"},
        )

    planner = ExperimentPlanner(cfg)
    runs = planner.plan()
    planner.write_snapshots(runs)
    ExperimentRunner(cfg, train_fn=train_fn, eval_fn=eval_fn).run_all(runs)
    paths = ResultAggregator(planner.root, baseline_variant="fixed_beta_0.05").write_all()
    (planner.root / "README_FIXTURE.txt").write_text(
        "SYNTHETIC CPU FIXTURE\n"
        "data_source=test_fixture\n"
        "NOT real DPO ablation results. For engineering demo only.\n"
        f"paths={json.dumps(paths, indent=2)}\n",
        encoding="utf-8",
    )
    print("ROOT", planner.root)
    print("TOP", sorted(p.name for p in planner.root.iterdir()))
    print("PLOTS", list((planner.root / "plots").glob("*.png")))


if __name__ == "__main__":
    main()
