"""消融实验共享类型。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    TRAIN_OK = "train_ok"
    EVAL_FAILED = "eval_failed"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    DRY_RUN = "dry_run"


@dataclass
class PlannedRun:
    """一次具体实验（变体 × seed）。"""

    run_id: str
    variant: str
    seed: int
    overrides: dict[str, Any] = field(default_factory=dict)
    hydra_overrides: list[str] = field(default_factory=list)
    run_dir: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunRecord:
    """单次运行状态与产物引用（与 Planner / EvaluationResult 对齐）。"""

    run_id: str
    variant: str
    seed: int
    status: str
    run_dir: str
    checkpoint: str | None = None
    train_exit_code: int | None = None
    eval_report: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    attempts: int = 0
    hydra_overrides: list[str] = field(default_factory=list)
    overrides: dict[str, Any] = field(default_factory=dict)
    algorithm: str | None = None
    task: str | None = None
    data_source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunRecord":
        return cls(
            run_id=str(data["run_id"]),
            variant=str(data.get("variant", "")),
            seed=int(data.get("seed", 0)),
            status=str(data.get("status", RunStatus.PENDING.value)),
            run_dir=str(data.get("run_dir", "")),
            checkpoint=data.get("checkpoint"),
            train_exit_code=data.get("train_exit_code"),
            eval_report=data.get("eval_report"),
            metrics=dict(data.get("metrics") or {}),
            error=data.get("error"),
            attempts=int(data.get("attempts", 0)),
            hydra_overrides=list(data.get("hydra_overrides") or []),
            overrides=dict(data.get("overrides") or {}),
            algorithm=data.get("algorithm"),
            task=data.get("task"),
            data_source=data.get("data_source"),
        )

    def save(self, path: Path | None = None) -> Path:
        out = path or (Path(self.run_dir) / "status.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: Path) -> "RunRecord":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
