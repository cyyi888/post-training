"""Shared utilities. 重依赖（torch / wandb）按需导入。"""

from .env import load_dotenv
from .logging_utils import get_logger, setup_logging

__all__ = [
    "load_dotenv",
    "same_seeds",
    "generate_response",
    "test_model_with_questions",
    "setup_logging",
    "get_logger",
    "prepare_runtime",
    "teardown_runtime",
    "init_wandb",
    "finish_wandb",
    "log_metrics",
    "resolve_report_to",
]


def __getattr__(name: str):
    if name == "same_seeds":
        from .seed import same_seeds

        return same_seeds
    if name in ("generate_response", "test_model_with_questions"):
        from .generation import generate_response, test_model_with_questions

        return {
            "generate_response": generate_response,
            "test_model_with_questions": test_model_with_questions,
        }[name]
    if name in ("prepare_runtime", "teardown_runtime"):
        from .runtime import prepare_runtime, teardown_runtime

        return {"prepare_runtime": prepare_runtime, "teardown_runtime": teardown_runtime}[name]
    if name in ("init_wandb", "finish_wandb", "log_metrics", "resolve_report_to"):
        from .wandb_utils import finish_wandb, init_wandb, log_metrics, resolve_report_to

        return {
            "init_wandb": init_wandb,
            "finish_wandb": finish_wandb,
            "log_metrics": log_metrics,
            "resolve_report_to": resolve_report_to,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
