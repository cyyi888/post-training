"""统一指标出口 MetricSink（P0）。

算法经适配器写入此处；后端目前复用 ``wandb_utils.log_metrics``，
不在此做多后端插件架构。
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger("posttrainlab.metrics")

# Step 语义（勿混用）
STEP_OPTIMIZER = "optimizer_step"
STEP_ROLLOUT = "rollout_step"
STEP_EPOCH = "epoch"
STEP_FINAL = "final"

LogFn = Callable[..., None]


def filter_scalar_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """只保留可序列化标量。"""
    return {k: v for k, v in metrics.items() if isinstance(v, (int, float, bool, str))}


def map_hf_trainer_logs(logs: dict[str, Any]) -> dict[str, Any]:
    """
    HF/TRL ``Trainer.log`` 字典 → 统一命名。

    只做键名映射，不改数值含义。未知键加 ``hf/`` 前缀以免 silently drop。
    """
    mapping = {
        "loss": "train/loss",
        "eval_loss": "eval/loss",
        "learning_rate": "train/learning_rate",
        "epoch": "train/epoch",
        "grad_norm": "train/grad_norm",
        "train_runtime": "train/runtime_sec",
        "train_samples_per_second": "train/samples_per_second",
        "train_steps_per_second": "train/steps_per_second",
    }
    out: dict[str, Any] = {}
    for k, v in logs.items():
        if not isinstance(v, (int, float, bool)):
            continue
        if k in mapping:
            out[mapping[k]] = v
        elif k.startswith(("train/", "eval/", "sft/", "dpo/", "grpo/", "ppo/", "final/", "rollout/")):
            out[k] = v
        else:
            out[f"hf/{k}"] = v
    return out


class MetricSink:
    """
    统一指标汇聚点。

    - 仅主进程写入后端（DDP rank0）
    - ``step_kind`` 标明 step 语义，不与数值 step 混写进指标键（避免污染曲线）
    - 后端默认走现有 ``log_metrics``（W&B）
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        is_main_process: bool = True,
        algorithm: str = "",
        backend_log: LogFn | None = None,
        record_history: bool = False,
    ) -> None:
        self.enabled = bool(enabled)
        self.is_main_process = bool(is_main_process)
        self.algorithm = str(algorithm or "")
        self._backend_log = backend_log
        self.record_history = bool(record_history)
        self.history: list[dict[str, Any]] = []

    def log(
        self,
        metrics: dict[str, Any] | None,
        *,
        step: int | None = None,
        step_kind: str | None = None,
    ) -> None:
        if not self.is_main_process or not metrics:
            return
        payload = filter_scalar_metrics(metrics)
        if not payload:
            return
        if self.algorithm and "meta/algorithm" not in payload:
            payload = {**payload, "meta/algorithm": self.algorithm}
        if step_kind:
            payload = {**payload, "meta/step_kind": str(step_kind)}

        if self.record_history:
            self.history.append({"step": step, "step_kind": step_kind, "metrics": dict(payload)})

        if not self.enabled:
            return
        log_fn = self._backend_log
        if log_fn is None:
            from src.utils.wandb_utils import log_metrics

            log_fn = log_metrics
        try:
            if step is None:
                log_fn(payload)
            else:
                log_fn(payload, step=int(step))
        except TypeError:
            log_fn(payload)
        except Exception as exc:
            logger.warning("MetricSink 上报失败: %s", exc)


def make_metric_sink_callback(sink: MetricSink | LogFn, *, algorithm: str = "sft"):
    """构造 HF ``TrainerCallback``，在 ``on_log`` 时转发到 MetricSink。"""
    from transformers import TrainerCallback

    class _MetricSinkCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
            if not logs:
                return
            mapped = map_hf_trainer_logs(dict(logs))
            if not mapped:
                return
            step = int(getattr(state, "global_step", 0) or 0)
            if isinstance(sink, MetricSink):
                sink.log(mapped, step=step, step_kind=STEP_OPTIMIZER)
            else:
                try:
                    sink(mapped, step=step, step_kind=STEP_OPTIMIZER)
                except TypeError:
                    sink(mapped, step=step)

    return _MetricSinkCallback()
