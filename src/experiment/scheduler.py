from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from omegaconf import OmegaConf


class AblationScheduler:
    """
    Expand experiment.variants into concrete configs and run a user callback.

    The callback signature is ``fn(cfg, variant_name) -> dict metrics``.
    """

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.exp = cfg.experiment
        self.results: list[dict[str, Any]] = []

    def iter_variants(self):
        for variant in self.exp.variants:
            base = OmegaConf.to_container(self.cfg, resolve=True)
            # Drop nested experiment to avoid recursive expansion
            base.pop("experiment", None)
            merged = OmegaConf.create(base)
            overrides = variant.get("overrides", {})
            flat = OmegaConf.to_container(overrides, resolve=True) or {}
            for key, value in flat.items():
                OmegaConf.update(merged, key, value, merge=True)
            yield variant.name, merged

    def run(self, train_fn: Callable[[Any, str], dict[str, Any]]) -> list[dict[str, Any]]:
        out_root = Path(self.cfg.output_dir) / "ablation" / self.exp.name
        out_root.mkdir(parents=True, exist_ok=True)

        for name, vcfg in self.iter_variants():
            print(f"\n=== Ablation variant: {name} ===")
            metrics = train_fn(vcfg, name)
            row = {"variant": name, **metrics}
            self.results.append(row)
            with (out_root / f"{name}.json").open("w", encoding="utf-8") as f:
                json.dump(row, f, indent=2, ensure_ascii=False)

        summary_path = out_root / "summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"Ablation summary → {summary_path}")
        return self.results
