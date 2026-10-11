"""标准化评测结果。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvaluationResult:
    """统一评测返回值（任务无关）。"""

    task: str
    checkpoint: str
    checkpoint_kind: str  # full | lora | base
    metrics: dict[str, float] = field(default_factory=dict)
    report_path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    # 本地 Mock 测试不覆盖真实权重正确性；GPU 验收前保持 pending
    validation_status: str = "GPU_PENDING"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_report(self, path: str | Path) -> Path:
        from src.evaluators.report import save_report

        out = save_report(self.to_dict(), path)
        self.report_path = str(out)
        return out
