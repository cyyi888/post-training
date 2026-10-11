"""ResultAggregator：消融结果汇总 → CSV / Markdown / 对比图。

统计约定：
- 均值：算术平均（``statistics.fmean``）
- 标准差：样本标准差（``statistics.stdev``，除以 n-1）；n=1 时 std 为空（不伪造 0）
- 仅 ``succeeded`` / ``skipped`` 且存在有限数值评测指标的 run 参与均值
- ``failed`` / ``eval_failed`` 保留在 results，不进均值
- Baseline：``delta_*`` 为绝对差；``pct_delta_*`` 为相对百分比（分母为 baseline 均值）
"""

from __future__ import annotations

import csv
import json
import logging
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("posttrainlab.experiment.aggregator")

# 可参与统计的状态（仍需有有效数值指标）
_OK_STATUSES = frozenset({"succeeded", "skipped"})
_FAIL_STATUSES = frozenset({"failed", "eval_failed"})
_NON_METRIC_KEYS = frozenset(
    {
        "status",
        "data_source",
        "note",
        "run_id",
        "variant",
        "seed",
        "train_loss",
        "skipped",
    }
)

RESULTS_FIELDS = [
    "run_id",
    "algorithm",
    "seed",
    "variant",
    "sweep_parameter",
    "sweep_value",
    "checkpoint",
    "task",
    "metric_name",
    "metric_value",
    "status",
    "data_source",
    "error",
    "eval_report",
]

SUMMARY_FIELDS = [
    "sweep_parameter",
    "sweep_value",
    "variant",
    "task",
    "metric_name",
    "n_usable",
    "n_failed",
    "n_missing",
    "mean",
    "std",
    "std_kind",
    "delta_vs_baseline",
    "pct_delta_vs_baseline",
    "is_baseline",
]


class ResultAggregator:
    """
    扫描 ``ablation/<name>/runs/*/status.json`` 与阶段 A ``eval_report.json``，
    生成 results.csv / summary.csv / report.md / PNG 对比图。

    不依赖 WandB；不重新跑模型评测。
    """

    def __init__(
        self,
        ablation_root: str | Path,
        *,
        baseline_variant: str | None = None,
        require_plots: bool = True,
    ):
        self.root = Path(ablation_root)
        self.baseline_variant = baseline_variant
        self.require_plots = require_plots
        self.rows: list[dict[str, Any]] = []  # 长表原始行
        self.meta: dict[str, Any] = {}
        self.plan_by_id: dict[str, dict[str, Any]] = {}
        self.plot_files: list[str] = []
        self.warnings: list[str] = []

    # ----- 加载 -----

    def _load_meta(self) -> dict[str, Any]:
        path = self.root / "experiment_meta.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        return {}

    def _load_plan(self) -> dict[str, dict[str, Any]]:
        path = self.root / "plan.json"
        if not path.is_file():
            return {}
        items = json.loads(path.read_text(encoding="utf-8"))
        return {str(x["run_id"]): x for x in items if isinstance(x, dict) and "run_id" in x}

    def collect(self) -> list[dict[str, Any]]:
        self.meta = self._load_meta()
        self.plan_by_id = self._load_plan()
        if self.baseline_variant is None:
            self.baseline_variant = self._infer_baseline()

        sweep_params = list(self.meta.get("sweep_parameters") or [])
        if not sweep_params and isinstance(self.meta.get("sweep"), dict):
            sweep_params = sorted(str(k) for k in self.meta["sweep"].keys())

        rows: list[dict[str, Any]] = []
        status_files = sorted(self.root.glob("runs/*/status.json"))
        for status_path in status_files:
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
            except Exception as exc:
                self.warnings.append(f"损坏 status: {status_path} ({exc})")
                continue

            run_id = str(status.get("run_id") or status_path.parent.name)
            planned = self.plan_by_id.get(run_id, {})
            overrides = dict(status.get("overrides") or planned.get("overrides") or {})
            sweep_parameter, sweep_value = self._pick_sweep(overrides, sweep_params)

            algorithm = (
                status.get("algorithm")
                or self.meta.get("algorithm")
                or self._algorithm_from_snapshot(status_path.parent)
            )
            task = status.get("task") or self.meta.get("eval_task")
            data_source = status.get("data_source") or (status.get("metrics") or {}).get(
                "data_source"
            )
            checkpoint = status.get("checkpoint")
            st = str(status.get("status") or "unknown")
            error = status.get("error")

            metrics: dict[str, Any] = {}
            report_path = status.get("eval_report")
            # 兼容未写 eval_report 字段但目录内有报告
            if not report_path:
                cand = status_path.parent / "eval_report.json"
                if cand.is_file():
                    report_path = str(cand)
            if report_path and Path(report_path).is_file():
                try:
                    payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
                    # EvaluationResult.to_dict() 形状
                    nested = payload.get("metrics")
                    if isinstance(nested, dict):
                        metrics.update(nested)
                    if payload.get("task"):
                        task = payload["task"]
                    if payload.get("checkpoint") and not checkpoint:
                        checkpoint = payload["checkpoint"]
                    if payload.get("data_source"):
                        data_source = payload["data_source"]
                    extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}
                    if extra.get("data_source") and not data_source:
                        data_source = extra["data_source"]
                except Exception as exc:
                    self.warnings.append(f"读取评测报告失败 {report_path}: {exc}")

            # status.metrics 中非训练噪声字段也可作为补充（评测失败时可能为空）
            for k, v in dict(status.get("metrics") or {}).items():
                if k in _NON_METRIC_KEYS:
                    continue
                if k not in metrics and isinstance(v, (int, float)) and not isinstance(v, bool):
                    metrics[k] = v

            base_cols = {
                "run_id": run_id,
                "algorithm": algorithm,
                "seed": status.get("seed", planned.get("seed")),
                "variant": status.get("variant") or planned.get("variant"),
                "sweep_parameter": sweep_parameter,
                "sweep_value": sweep_value,
                "checkpoint": checkpoint,
                "task": task,
                "status": st,
                "data_source": data_source,
                "error": error,
                "eval_report": report_path,
            }

            numeric = {
                k: float(v)
                for k, v in metrics.items()
                if k not in _NON_METRIC_KEYS
                and isinstance(v, (int, float))
                and not isinstance(v, bool)
                and _finite(v)
            }
            if numeric:
                for name, value in sorted(numeric.items()):
                    row = dict(base_cols)
                    row["metric_name"] = name
                    row["metric_value"] = value
                    rows.append(row)
            else:
                # 失败 / 缺失：保留一行，metric 为空，不能伪装成功
                row = dict(base_cols)
                row["metric_name"] = None
                row["metric_value"] = None
                rows.append(row)

        self.rows = rows
        return rows

    def _infer_baseline(self) -> str | None:
        """取 sweep 第一档对应 variant 为 baseline（若可解析）。"""
        sweep = self.meta.get("sweep") or {}
        if not isinstance(sweep, dict) or not sweep:
            return None
        key = sorted(sweep.keys())[0]
        vals = sweep[key]
        if isinstance(vals, (list, tuple)) and vals:
            first = vals[0]
            short = str(key).split(".")[-1]
            return f"{short}_{first}"
        return None

    @staticmethod
    def _pick_sweep(
        overrides: dict[str, Any], sweep_params: list[str]
    ) -> tuple[str | None, Any]:
        for key in sweep_params:
            if key in overrides:
                return key, overrides[key]
        # 回退：除 seed / baseline 固定项外的第一个
        for key, val in sorted(overrides.items()):
            if key == "seed":
                continue
            if key.startswith("training.") and key not in (
                "training.loss_type",
                "training.beta_mode",
                "training.penalty.enabled",
            ):
                return key, val
        return None, None

    @staticmethod
    def _algorithm_from_snapshot(run_dir: Path) -> str | None:
        snap = run_dir / "config_snapshot.json"
        if not snap.is_file():
            return None
        try:
            data = json.loads(snap.read_text(encoding="utf-8"))
            t = data.get("training") or {}
            return t.get("algorithm")
        except Exception:
            return None

    # ----- 统计 -----

    def summary_rows(self) -> list[dict[str, Any]]:
        """长表汇总：每 (sweep_value/variant, task, metric) 一行。"""
        if not self.rows:
            self.collect()

        # group key
        groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
        for row in self.rows:
            key = (
                row.get("sweep_parameter"),
                row.get("sweep_value"),
                row.get("variant"),
                row.get("task") or "unknown",
                row.get("metric_name"),
            )
            groups[key].append(row)

        # 也需要按 variant+task 统计失败（无 metric 的行）
        status_groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
        seen_runs: dict[tuple, set[str]] = defaultdict(set)
        for row in self.rows:
            vk = (row.get("variant"), row.get("task") or "unknown")
            rid = str(row.get("run_id"))
            if rid in seen_runs[vk]:
                continue
            seen_runs[vk].add(rid)
            status_groups[vk].append(row)

        summaries: list[dict[str, Any]] = []
        # 只对有 metric_name 的组做 mean；失败行 metric_name=None 单独处理覆盖率
        for key, items in sorted(groups.items(), key=lambda x: tuple(str(i) for i in x[0])):
            sweep_parameter, sweep_value, variant, task, metric_name = key
            if metric_name is None:
                continue

            usable_vals: list[float] = []
            usable_runs: set[str] = set()
            for r in items:
                if str(r.get("status")) not in _OK_STATUSES:
                    continue
                val = r.get("metric_value")
                if isinstance(val, (int, float)) and _finite(val):
                    usable_vals.append(float(val))
                    usable_runs.add(str(r.get("run_id")))

            vk = (variant, task)
            status_items = status_groups.get(vk, [])
            n_failed = sum(1 for r in status_items if str(r.get("status")) in _FAIL_STATUSES)
            n_missing = sum(
                1
                for r in status_items
                if str(r.get("status")) not in _OK_STATUSES | _FAIL_STATUSES
            )

            entry: dict[str, Any] = {
                "sweep_parameter": sweep_parameter,
                "sweep_value": sweep_value,
                "variant": variant,
                "task": task,
                "metric_name": metric_name,
                "n_usable": len(usable_vals),
                "n_failed": n_failed,
                "n_missing": n_missing,
                "mean": None,
                "std": None,
                "std_kind": "sample_stdev(n-1)",
                "delta_vs_baseline": None,
                "pct_delta_vs_baseline": None,
                "is_baseline": False,
            }
            if usable_vals:
                entry["mean"] = float(statistics.fmean(usable_vals))
                if len(usable_vals) >= 2:
                    entry["std"] = float(statistics.stdev(usable_vals))
                else:
                    entry["std"] = None  # n=1 不伪造标准差
            summaries.append(entry)

        # baseline 对比（同 task + metric_name）
        if self.baseline_variant:
            by_tm: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
            for s in summaries:
                by_tm[(str(s["task"]), str(s["metric_name"]))][str(s["variant"])] = s
            for (_task, _metric), var_map in by_tm.items():
                base = var_map.get(str(self.baseline_variant))
                if not base or base.get("mean") is None:
                    continue
                base["is_baseline"] = True
                bmean = float(base["mean"])
                for variant, s in var_map.items():
                    if variant == self.baseline_variant:
                        continue
                    s["is_baseline"] = False
                    if s.get("mean") is None:
                        continue
                    delta = float(s["mean"]) - bmean
                    s["delta_vs_baseline"] = delta
                    if abs(bmean) > 1e-12:
                        s["pct_delta_vs_baseline"] = 100.0 * delta / bmean
                    else:
                        s["pct_delta_vs_baseline"] = None
        return summaries

    # ----- 写出 -----

    def write_results_csv(self, path: Path | None = None) -> Path:
        if not self.rows:
            self.collect()
        out = path or (self.root / "results.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=RESULTS_FIELDS, extrasaction="ignore")
            w.writeheader()
            for row in self.rows:
                w.writerow({k: row.get(k) for k in RESULTS_FIELDS})
        return out

    def write_summary_csv(self, path: Path | None = None) -> Path:
        summaries = self.summary_rows()
        out = path or (self.root / "summary.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
            w.writeheader()
            for row in summaries:
                w.writerow({k: row.get(k) for k in SUMMARY_FIELDS})
        return out

    def write_plots(self, path: Path | None = None) -> Path:
        """
        每 (task, metric) 一张图；X=sweep_value（数值）或 variant；Y=mean±样本标准差。
        无有效数据时不绘图并写入 warnings。
        matplotlib 缺失时：require_plots=True 则抛错，否则警告。
        """
        summaries = self.summary_rows()
        out_dir = Path(path) if path is not None else (self.root / "plots")
        out_dir.mkdir(parents=True, exist_ok=True)
        self.plot_files = []

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as exc:
            msg = (
                "matplotlib 未安装，无法生成对比图。请执行: "
                "pip install matplotlib  （或 pip install -e \".[dev]\"）"
            )
            self.warnings.append(msg)
            logger.error(msg)
            if self.require_plots:
                raise RuntimeError(msg) from exc
            return out_dir

        by_tm: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for s in summaries:
            if s.get("mean") is None:
                continue
            by_tm[(str(s["task"]), str(s["metric_name"]))].append(s)

        if not by_tm:
            self.warnings.append("无有效评测均值，跳过绘图（避免误导性空图）")
            (out_dir / "README.txt").write_text(
                "No plots: no usable metric means.\n", encoding="utf-8"
            )
            return out_dir

        for (task, metric), items in sorted(by_tm.items()):
            # 优先按数值 sweep_value 排序
            def _sort_key(it: dict[str, Any]):
                sv = it.get("sweep_value")
                try:
                    return (0, float(sv))
                except (TypeError, ValueError):
                    return (1, str(it.get("variant")))

            items = sorted(items, key=_sort_key)
            xs_num: list[float] = []
            labels: list[str] = []
            means: list[float] = []
            stds: list[float] = []
            use_numeric_x = True
            for it in items:
                try:
                    xs_num.append(float(it["sweep_value"]))
                except (TypeError, ValueError):
                    use_numeric_x = False
                labels.append(str(it.get("variant")))
                means.append(float(it["mean"]))
                std = it.get("std")
                stds.append(float(std) if isinstance(std, (int, float)) else 0.0)

            fig, ax = plt.subplots(figsize=(7, 4.5))
            sweep_p = items[0].get("sweep_parameter") or "sweep"
            if use_numeric_x and len(xs_num) == len(means):
                ax.errorbar(
                    xs_num,
                    means,
                    yerr=stds,
                    fmt="o-",
                    capsize=4,
                    linewidth=1.5,
                    markersize=7,
                )
                ax.set_xlabel(str(sweep_p))
                ax.set_xticks(xs_num)
            else:
                xpos = list(range(len(labels)))
                ax.bar(xpos, means, yerr=stds, capsize=4, width=0.6)
                ax.set_xticks(xpos)
                ax.set_xticklabels(labels, rotation=25, ha="right")
                ax.set_xlabel("variant")

            ax.set_ylabel(f"{task} / {metric}")
            ax.set_title(f"Ablation: {sweep_p} vs {task}.{metric}")
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            out = out_dir / f"{task}_{metric}.png"
            fig.savefig(out, dpi=120)
            plt.close(fig)
            if out.stat().st_size <= 0:
                raise RuntimeError(f"绘图文件为空: {out}")
            self.plot_files.append(str(out))

        return out_dir

    def write_report_md(self, path: Path | None = None) -> Path:
        if not self.rows:
            self.collect()
        summaries = self.summary_rows()
        out = path or (self.root / "report.md")

        fixture = any(str(r.get("data_source")) == "test_fixture" for r in self.rows)
        n_fail = len({r["run_id"] for r in self.rows if str(r.get("status")) in _FAIL_STATUSES})
        n_ok = len({r["run_id"] for r in self.rows if str(r.get("status")) in _OK_STATUSES})
        n_missing = len(
            {
                r["run_id"]
                for r in self.rows
                if str(r.get("status")) not in _OK_STATUSES | _FAIL_STATUSES
            }
        )

        lines: list[str] = [
            "# Ablation Report",
            "",
        ]
        if fixture:
            lines += [
                "> **WARNING**: `data_source=test_fixture`",
                "> 本报告仅用于**功能/集成测试**，指标为模拟数据，**不能**作为真实算法效果结论。",
                "",
            ]

        lines += [
            "## 实验信息",
            "",
            f"- 实验名称: `{self.meta.get('experiment_name') or self.root.name}`",
            f"- 算法: `{self.meta.get('algorithm')}`",
            f"- 模型: `{self.meta.get('model_name')}`",
            f"- 初始化 Checkpoint: `{self.meta.get('init_from_checkpoint')}`",
            f"- 数据集: `{self.meta.get('dataset')}`",
            f"- 数据版本: `{self.meta.get('data_version')}`",
            f"- 消融参数: `{self.meta.get('sweep')}`",
            f"- 固定超参数 (baseline_overrides): `{self.meta.get('baseline_overrides')}`",
            f"- Seed 列表: `{self.meta.get('seeds')}`",
            f"- 记录时间 (UTC): `{self.meta.get('recorded_at')}`",
            f"- 报告生成时间 (UTC): `{datetime.now(timezone.utc).isoformat()}`",
            f"- 可复现配置: `{self.root / 'plan.json'}` / `{self.root / 'experiment_meta.json'}`",
            f"- Ablation root: `{self.root}`",
            "",
            "## 统计说明",
            "",
            "- 均值：算术平均。",
            "- 标准差：样本标准差（`statistics.stdev`，分母 n-1）。",
            "- 有效样本数 n=1 时不报告标准差（单元格为空），避免伪造。",
            "- 仅状态为 `succeeded`/`skipped` 且指标为有限数值的 run 参与均值。",
            "- `delta_vs_baseline` 为绝对差；`pct_delta_vs_baseline` 为相对百分比。",
            "- **不做**统计显著性检验；多 seed 均值比较≠显著结论。",
            "",
            "## 覆盖率",
            "",
            f"- 成功/跳过 run 数: **{n_ok}**",
            f"- 失败 run 数 (train/eval): **{n_fail}**",
            f"- 缺失/未完成 (dry_run/pending/...): **{n_missing}**",
            f"- 原始结果行数 (长表): **{len(self.rows)}**",
            "",
            "## 消融汇总表",
            "",
        ]

        if not summaries:
            lines.append("_无有效汇总（可能全部失败或缺失）。_")
        else:
            header = [
                "sweep_value",
                "variant",
                "task",
                "metric",
                "n",
                "mean",
                "std",
                "Δbaseline",
                "%Δ",
            ]
            lines.append("| " + " | ".join(header) + " |")
            lines.append("| " + " | ".join("---" for _ in header) + " |")
            for s in summaries:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            _fmt(s.get("sweep_value")),
                            str(s.get("variant")),
                            str(s.get("task")),
                            str(s.get("metric_name")),
                            str(s.get("n_usable")),
                            _fmt_float(s.get("mean")),
                            _fmt_float(s.get("std")),
                            _fmt_float(s.get("delta_vs_baseline")),
                            _fmt_float(s.get("pct_delta_vs_baseline")),
                        ]
                    )
                    + " |"
                )

        lines += [
            "",
            "## 原始结果（节选）",
            "",
            "完整数据见 `results.csv`。可通过 `run_id` 关联 `runs/<run_id>/status.json` "
            "与 `eval_report.json`。",
            "",
        ]
        # 最多展示 20 行
        preview = self.rows[:20]
        if preview:
            ph = ["run_id", "seed", "sweep_value", "task", "metric_name", "metric_value", "status"]
            lines.append("| " + " | ".join(ph) + " |")
            lines.append("| " + " | ".join("---" for _ in ph) + " |")
            for r in preview:
                lines.append(
                    "| "
                    + " | ".join(_fmt(r.get(k)) for k in ph)
                    + " |"
                )
            if len(self.rows) > 20:
                lines.append("")
                lines.append(f"_… 其余 {len(self.rows) - 20} 行见 results.csv_")

        lines += ["", "## 可视化", ""]
        if self.plot_files:
            lines.append(
                "下图横轴为消融变量（或 variant），纵轴为任务指标均值；"
                "误差线为多样本时的**样本标准差**（n=1 时误差为 0 仅作占位，表中 std 为空）。"
            )
            lines.append("")
            for pf in self.plot_files:
                rel = Path(pf).name
                lines.append(f"![{rel}](plots/{rel})")
                lines.append("")
        else:
            lines.append("_未生成图表。_")
            for w in self.warnings:
                if "图" in w or "plot" in w.lower() or "matplotlib" in w.lower():
                    lines.append(f"- {w}")

        lines += [
            "",
            "## 实验限制",
            "",
            f"- 失败实验数: {n_fail}",
            f"- 缺失/未完成: {n_missing}",
            f"- 数据来源含 test_fixture: **{fixture}**",
            "- 真实模型训练与评测正确性: **GPU_PENDING**（见 `docs/gpu_validation.md`）",
            "",
        ]
        if self.warnings:
            lines.append("### 生成警告")
            lines.append("")
            for w in self.warnings:
                lines.append(f"- {w}")
            lines.append("")

        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return out

    def write_all(self) -> dict[str, str]:
        """幂等重跑：重新 collect，覆盖输出文件，不修改 runs/*/status.json。"""
        self.warnings = []
        self.collect()
        paths: dict[str, str] = {
            "results_csv": str(self.write_results_csv()),
            "summary_csv": str(self.write_summary_csv()),
        }
        plots_dir = self.write_plots()
        paths["plots_dir"] = str(plots_dir)
        paths["report_md"] = str(self.write_report_md())
        overview = {
            "root": str(self.root),
            "n_result_rows": len(self.rows),
            "n_plots": len(self.plot_files),
            "baseline_variant": self.baseline_variant,
            "paths": paths,
            "warnings": list(self.warnings),
            "note": "If any row has data_source=test_fixture, numbers are synthetic.",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        overview_path = self.root / "aggregate_overview.json"
        overview_path.write_text(
            json.dumps(overview, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        paths["overview"] = str(overview_path)
        return paths


def _finite(v: Any) -> bool:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return f == f and f not in (float("inf"), float("-inf"))


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    return str(v)


def _fmt_float(v: Any) -> str:
    if v is None:
        return ""
    try:
        return f"{float(v):.6g}"
    except (TypeError, ValueError):
        return str(v)
