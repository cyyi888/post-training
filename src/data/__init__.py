"""数据层：统一 Sample、适配器、清洗、划分、版本、Chat Template、流水线。"""

from .adapters import get_adapter
from .chat_templates import apply_chat_template_config, render_chat
from .cleaning import clean_samples
from .schema import Sample
from .split import split_samples
from .versioning import (
    DatasetVersion,
    compute_samples_hash,
    list_versions,
    load_version,
    make_version,
    save_version,
    verify_version,
)

# 依赖 datasets / 较重组件：按需导入，避免纯单测环境强依赖
try:
    from .loaders import load_preference_dataset, load_task_dataset
    from .pipeline import build_pipeline, build_samples
except ImportError:  # pragma: no cover
    load_preference_dataset = None  # type: ignore[assignment]
    load_task_dataset = None  # type: ignore[assignment]
    build_pipeline = None  # type: ignore[assignment]
    build_samples = None  # type: ignore[assignment]

__all__ = [
    "Sample",
    "get_adapter",
    "clean_samples",
    "split_samples",
    "build_pipeline",
    "build_samples",
    "load_task_dataset",
    "load_preference_dataset",
    "apply_chat_template_config",
    "render_chat",
    "DatasetVersion",
    "make_version",
    "save_version",
    "load_version",
    "verify_version",
    "list_versions",
    "compute_samples_hash",
]
