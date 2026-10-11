"""Task evaluation, metrics, and report generation."""

from src.evaluators.checkpoint import CheckpointError, detect_checkpoint_kind, resolve_checkpoint
from src.evaluators.factory import available_tasks, build_evaluator
from src.evaluators.gsm8k_eval import GSM8KEvaluator
from src.evaluators.humaneval_eval import HumanEvalEvaluator
from src.evaluators.report import save_report
from src.evaluators.result import EvaluationResult
from src.evaluators.service import evaluate, resolve_eval_task

__all__ = [
    "CheckpointError",
    "EvaluationResult",
    "GSM8KEvaluator",
    "HumanEvalEvaluator",
    "available_tasks",
    "build_evaluator",
    "detect_checkpoint_kind",
    "evaluate",
    "resolve_checkpoint",
    "resolve_eval_task",
    "save_report",
]
