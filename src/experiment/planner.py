"""ExperimentPlanner：展开消融组合、分配稳定 Run ID、落盘配置快照。"""

from __future__ import annotations

import hashlib
import json
import logging
from itertools import product
from pathlib import Path
from typing import Any, Iterable

from omegaconf import OmegaConf

from src.experiment.types import PlannedRun

logger = logging.getLogger("posttrainlab.experiment.planner")


def _exp_section(cfg: Any) -> dict[str, Any]:
    exp = cfg.get("experiment") if hasattr(cfg, "get") else getattr(cfg, "experiment", None)
    if exp is None:
        return {}
    if isinstance(exp, dict):
        return dict(exp)
    try:
        return dict(OmegaConf.to_container(exp, resolve=True) or {})
    except Exception:
        return {}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def format_hydra_override(key: str, value: Any) -> str:
    """把 (dot.key, value) 编成 Hydra CLI override。"""
    if value is None:
        return f"{key}=null"
    if isinstance(value, bool):
        return f"{key}={str(value).lower()}"
    if isinstance(value, (int, float)):
        return f"{key}={value}"
    text = str(value)
    # 含特殊字符时加引号
    if any(c in text for c in " ,=[]{}"):
        return f'{key}="{text}"'
    return f"{key}={text}"


def stable_run_id(experiment: str, variant: str, seed: int, overrides: dict[str, Any]) -> str:
    """稳定 Run ID：实验名 + 变体 + seed + 排序后的 overrides 哈希。"""
    payload = {
        "experiment": experiment,
        "variant": variant,
        "seed": int(seed),
        "overrides": {k: overrides[k] for k in sorted(overrides)},
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    safe_variant = "".join(c if c.isalnum() or c in "-_" else "_" for c in variant)
    return f"{safe_variant}-s{seed}-{digest}"


class ExperimentPlanner:
    """
    根据 ``experiment`` 配置生成全部 PlannedRun。

    支持：
    - ``variants`` 列表（显式 overrides）
    - ``sweep`` 单变量/多变量笛卡尔积（如 ``training.fixed_beta: [0.1,0.5,1.0]``）
    - ``seeds`` 多 seed
    """

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.exp = _exp_section(cfg)
        self.name = str(self.exp.get("name") or "ablation")
        out = cfg.get("output_dir", "outputs") if hasattr(cfg, "get") else "outputs"
        self.root = Path(str(out)) / "ablation" / self.name

    def seeds(self) -> list[int]:
        seeds = self.exp.get("seeds")
        if seeds in (None, [], ""):
            base_seed = int(self.cfg.get("seed", 0) if hasattr(self.cfg, "get") else 0)
            return [base_seed]
        return [int(s) for s in _as_list(seeds)]

    def iter_variant_overrides(self) -> Iterable[tuple[str, dict[str, Any]]]:
        """产出 (variant_name, override_dict)。"""
        variants = self.exp.get("variants") or []
        sweep = self.exp.get("sweep") or {}

        if variants:
            for variant in variants:
                if isinstance(variant, dict):
                    name = str(variant.get("name") or "variant")
                    overrides = dict(variant.get("overrides") or {})
                else:
                    name = str(variant.get("name"))
                    ov = variant.get("overrides", {})
                    overrides = dict(OmegaConf.to_container(ov, resolve=True) or {})
                yield name, {str(k): v for k, v in overrides.items()}
            return

        baseline = dict(self.exp.get("baseline_overrides") or {})

        if sweep:
            # sweep: {key: [v1, v2]} → 笛卡尔积；单 key 时变体名用 key=value
            keys = sorted(str(k) for k in sweep.keys())
            value_lists = [_as_list(sweep[k]) for k in keys]
            for values in product(*value_lists):
                overrides = dict(baseline)
                overrides.update({k: v for k, v in zip(keys, values)})
                if len(keys) == 1:
                    name = f"{keys[0].split('.')[-1]}_{values[0]}"
                else:
                    name = "_".join(f"{k.split('.')[-1]}={v}" for k, v in zip(keys, values))
                # 保留小数点，便于 baseline_variant=fixed_beta_0.05 对齐
                safe = "".join(c if c.isalnum() or c in "-_.=+" else "_" for c in name)
                yield safe, overrides
            return

        # 无 variants/sweep：单 baseline
        yield "baseline", dict(baseline)

    def plan(self) -> list[PlannedRun]:
        runs: list[PlannedRun] = []
        for variant, overrides in self.iter_variant_overrides():
            for seed in self.seeds():
                full_overrides = dict(overrides)
                full_overrides["seed"] = int(seed)
                run_id = stable_run_id(self.name, variant, int(seed), full_overrides)
                run_dir = self.root / "runs" / run_id
                hydra = [format_hydra_override(k, v) for k, v in sorted(full_overrides.items())]
                # 每个 run 独立 output 子目录，避免互相覆盖 checkpoint
                hydra.append(format_hydra_override("output_dir", str(run_dir / "outputs")))
                runs.append(
                    PlannedRun(
                        run_id=run_id,
                        variant=variant,
                        seed=int(seed),
                        overrides=full_overrides,
                        hydra_overrides=hydra,
                        run_dir=str(run_dir),
                    )
                )
        logger.info("Planner: experiment=%s runs=%s", self.name, len(runs))
        return runs

    def write_snapshots(self, runs: list[PlannedRun] | None = None) -> Path:
        """写入 plan.json、experiment_meta.json 与各 run 的 config_snapshot.json。"""
        from datetime import datetime, timezone

        planned = runs if runs is not None else self.plan()
        self.root.mkdir(parents=True, exist_ok=True)
        plan_path = self.root / "plan.json"
        plan_path.write_text(
            json.dumps([r.to_dict() for r in planned], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        base = OmegaConf.to_container(self.cfg, resolve=True)
        if isinstance(base, dict):
            base = dict(base)
            exp_cfg = base.pop("experiment", None)
        else:
            exp_cfg = None
            base = {}

        training = base.get("training") if isinstance(base, dict) else {}
        model = base.get("model") if isinstance(base, dict) else {}
        data = base.get("data") if isinstance(base, dict) else {}
        sweep = dict(self.exp.get("sweep") or {})
        meta = {
            "experiment_name": self.name,
            "description": self.exp.get("description"),
            "algorithm": (training or {}).get("algorithm") if isinstance(training, dict) else None,
            "model_name": (model or {}).get("name") if isinstance(model, dict) else None,
            "init_from_checkpoint": None,
            "dataset": (data or {}).get("name") if isinstance(data, dict) else None,
            "data_version": base.get("data_version") if isinstance(base, dict) else None,
            "sweep": sweep,
            "sweep_parameters": sorted(str(k) for k in sweep.keys()),
            "baseline_overrides": dict(self.exp.get("baseline_overrides") or {}),
            "seeds": self.seeds(),
            "eval_task": self.exp.get("eval_task"),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "n_planned": len(planned),
        }
        eng = base.get("engineering") if isinstance(base, dict) else None
        if isinstance(eng, dict):
            meta["init_from_checkpoint"] = eng.get("init_from_checkpoint")
        (self.root / "experiment_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        if exp_cfg is not None:
            (self.root / "experiment_config.json").write_text(
                json.dumps(exp_cfg, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )

        for run in planned:
            rdir = Path(run.run_dir)
            rdir.mkdir(parents=True, exist_ok=True)
            merged = OmegaConf.create(base)
            for key, value in run.overrides.items():
                OmegaConf.update(merged, key, value, merge=True)
            snap = rdir / "config_snapshot.json"
            snap.write_text(
                json.dumps(OmegaConf.to_container(merged, resolve=True), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        return plan_path
