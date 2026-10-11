#!/usr/bin/env python
"""消融实验入口（Hydra）：ExperimentPlanner + ExperimentRunner + ResultAggregator。

常用：
  python scripts/ablation.py                              # 默认 dry_run
  python scripts/ablation.py experiment.dry_run=false     # GPU 真实训练
  python scripts/ablation.py experiment.report_only=true  # 只根据已有 runs 重生成报告
  python scripts/ablation.py experiment.eval_only=true    # 仅重试评测（保留 checkpoint）
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment import ExperimentPlanner, ExperimentRunner
from src.experiment.aggregator import ResultAggregator
from src.utils.runtime import prepare_runtime, teardown_runtime

logger = logging.getLogger("posttrainlab.ablation")


@hydra.main(version_base=None, config_path="../configs", config_name="experiment/ablation")
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    prepare_runtime(cfg, init_wb=bool(cfg.experiment.get("track_wandb", False)))
    try:
        report_only = bool(cfg.experiment.get("report_only", False))
        dry = bool(cfg.experiment.get("dry_run", True))
        logger.info(
            "Ablation: %s dry_run=%s report_only=%s eval_only=%s seeds=%s",
            cfg.experiment.name,
            dry,
            report_only,
            bool(cfg.experiment.get("eval_only", False)),
            list(cfg.experiment.get("seeds") or []),
        )
        planner = ExperimentPlanner(cfg)
        runs = planner.plan()
        planner.write_snapshots(runs)
        logger.info("Planned %s runs → %s", len(runs), planner.root / "plan.json")

        if not report_only:
            runner = ExperimentRunner(cfg)
            records = runner.run_all(runs)
            n_dry = sum(1 for r in records if r.status == "dry_run")
            n_ok = sum(1 for r in records if r.status in ("succeeded", "skipped"))
            n_fail = sum(1 for r in records if r.status in ("failed", "eval_failed"))
            logger.info(
                "Ablation finished total=%s dry_run=%s ok/skip=%s fail=%s",
                len(records),
                n_dry,
                n_ok,
                n_fail,
            )
        else:
            logger.info("report_only：跳过训练/评测，仅聚合 %s", planner.root)

        if bool(cfg.experiment.get("aggregate", True)):
            try:
                baseline = cfg.experiment.get("baseline_variant")
                agg = ResultAggregator(
                    planner.root,
                    baseline_variant=str(baseline) if baseline not in (None, "", "null") else None,
                    require_plots=bool(cfg.experiment.get("require_plots", True)),
                )
                paths = agg.write_all()
                logger.info("Reports: %s", paths)
            except Exception as exc:
                # 报告失败不得抹掉已有训练/评测产物
                logger.exception("报告生成失败（训练/评测产物保留）: %s", exc)
                raise
    finally:
        teardown_runtime()


if __name__ == "__main__":
    main()
