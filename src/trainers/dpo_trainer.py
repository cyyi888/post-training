"""DPO 训练：基于 TRL DPOTrainer（底层 Accelerate）扩展自研 Loss / 动态 β / WandB。

WandB：
- ``BaseTrainer.prepare_engineering`` 调用 ``init_wandb``；
- HF ``report_to`` 固定 ``none``，经 MetricSink / ``metrics_fn`` 上报，避免重复；
- ``CustomDPOTrainer`` 上报：``train/loss``、``train/mean_kl``、``train/beta``、
  ``val/loss``、``val/mean_kl``。
"""

from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path
from typing import Any

from src.algorithms.dpo_losses import DPOLossConfig, compute_preference_loss
from src.algorithms.dynamic_beta import DynamicBetaController
from src.core.base import BaseTrainer, TrainContext, TrainResult
from src.core.registry import ALGORITHM_REGISTRY, register

logger = logging.getLogger("posttrainlab.dpo")

BETA_STATE_FILENAME = "dynamic_beta_state.json"


def _as_dict(section: Any) -> dict[str, Any]:
    if section is None:
        return {}
    if isinstance(section, dict):
        return dict(section)
    if hasattr(section, "items"):
        try:
            return dict(section.items())
        except Exception:
            pass
    data = getattr(section, "__dict__", None)
    if isinstance(data, dict):
        return {k: v for k, v in data.items() if not str(k).startswith("_")}
    return {}


def _filter_kwargs(fn: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    params = inspect.signature(fn).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def _resolve_beta_mode(tcfg: Any) -> str:
    mode = str(tcfg.get("beta_mode", "fixed")).lower()
    if mode not in ("fixed", "dynamic"):
        raise ValueError(f"training.beta_mode 必须是 fixed|dynamic，收到: {mode!r}")
    return mode


def _resolve_fixed_beta(tcfg: Any) -> float:
    if tcfg.get("fixed_beta") is not None:
        return float(tcfg.get("fixed_beta"))
    return 0.1


def _resolve_dynamic_beta_section(tcfg: Any) -> dict[str, Any]:
    dyn = tcfg.get("dynamic_beta")
    if isinstance(dyn, bool):
        raise ValueError(
            "training.dynamic_beta 必须是嵌套表（init_beta/target_kl/...），"
            "不再接受 bool；请改用 training.beta_mode=fixed|dynamic"
        )
    if tcfg.get("beta") is not None:
        logger.warning(
            "已忽略旧字段 training.beta=%s；请改用 training.fixed_beta / "
            "training.dynamic_beta.init_beta",
            tcfg.get("beta"),
        )
    return _as_dict(dyn)


def _build_loss_cfg(tcfg: Any) -> DPOLossConfig:
    pen = _as_dict(tcfg.get("penalty"))
    dyn = _resolve_dynamic_beta_section(tcfg)
    beta_mode = _resolve_beta_mode(tcfg)
    fixed_beta = _resolve_fixed_beta(tcfg)
    init_beta = float(dyn.get("init_beta", fixed_beta))
    return DPOLossConfig(
        loss_type=str(tcfg.get("loss_type", "dpo")),
        beta=init_beta if beta_mode == "dynamic" else fixed_beta,
        beta_mode=beta_mode,
        beta_min=float(dyn.get("min_beta", 0.01)),
        beta_max=float(dyn.get("max_beta", 1.0)),
        target_kl=float(dyn.get("target_kl", 4.0)),
        adjust_rate=float(dyn.get("adjust_rate", 0.1)),
        margin=float(tcfg.get("margin", 1.0)),
        label_smoothing=float(tcfg.get("label_smoothing", 0.0)),
        penalty_enabled=bool(pen.get("enabled", False)),
        penalty_type=str(pen.get("type", "none")),
        penalty_weight=float(pen.get("weight", 0.0)),
        reference_free=bool(tcfg.get("reference_free", False)),
    )


def _build_beta_controller(cfg: DPOLossConfig) -> DynamicBetaController | None:
    if cfg.resolved_beta_mode() != "dynamic":
        return None
    return DynamicBetaController(
        init_beta=cfg.beta,
        target_kl=cfg.target_kl,
        adjust_rate=cfg.adjust_rate,
        min_beta=cfg.beta_min,
        max_beta=cfg.beta_max,
    )


class CustomDPOTrainer:
    """
    TRL ``DPOTrainer`` 子类工厂。

    实际类在 ``build_custom_dpo_trainer_cls`` 中动态创建，避免未安装 trl 时导入失败。
    """


def build_custom_dpo_trainer_cls():
    """返回继承自 TRL DPOTrainer 的 CustomDPOTrainer 类。"""
    from trl import DPOTrainer

    class _CustomDPOTrainer(DPOTrainer):
        """自研 Loss + 动态 β + 额外 WandB 指标。"""

        def __init__(
            self,
            *args: Any,
            custom_loss_type: str = "dpo",
            beta_mode: str = "fixed",
            margin: float = 1.0,
            beta_controller: DynamicBetaController | None = None,
            metrics_fn: Any | None = None,
            **kwargs: Any,
        ):
            self.custom_loss_type = str(custom_loss_type).lower()
            self.beta_mode = str(beta_mode).lower()
            self.margin = float(margin)
            self.beta_controller = beta_controller
            self.metrics_fn = metrics_fn
            self._last_train_kl = 0.0
            self._last_eval_kl = 0.0
            self._last_eval_loss = 0.0
            super().__init__(*args, **kwargs)
            # 统一用控制器 / 配置中的 beta
            if self.beta_controller is not None:
                self.beta = float(self.beta_controller.beta)
                if hasattr(self, "args") and hasattr(self.args, "beta"):
                    self.args.beta = self.beta
            self._freeze_reference_model()

        def _emit_metrics(self, payload: dict[str, Any], *, step: int | None = None) -> None:
            """结构化指标走 MetricSink；不经 HF WandbCallback。"""
            fn = self.metrics_fn
            if fn is None:
                return
            try:
                fn(payload, step=step, step_kind="optimizer_step")
            except TypeError:
                try:
                    fn(payload, step=step)
                except TypeError:
                    fn(payload)

        def _freeze_reference_model(self) -> None:
            ref = getattr(self, "ref_model", None)
            if ref is None:
                return
            ref.eval()
            for p in ref.parameters():
                p.requires_grad = False

        def get_batch_loss_metrics(
            self,
            model,
            batch: dict[str, Any],
            train_eval: str = "train",
        ):
            """用自研 DPO/IPO 替换 TRL 原生损失，并附加 mean_kl / beta。"""
            metrics: dict[str, float] = {}
            model_output = self.concatenated_forward(model, batch)

            if "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
                ref_chosen_logps = batch["ref_chosen_logps"]
                ref_rejected_logps = batch["ref_rejected_logps"]
            else:
                ref_chosen_logps, ref_rejected_logps = self.compute_ref_log_probs(batch)

            loss, loss_metrics = compute_preference_loss(
                self.custom_loss_type,
                model_output["chosen_logps"],
                model_output["rejected_logps"],
                ref_chosen_logps,
                ref_rejected_logps,
                beta=float(self.beta),
                margin=self.margin,
            )

            prefix = "eval_" if train_eval == "eval" else ""
            mean_kl = float(loss_metrics["mean_kl"])
            metrics[f"{prefix}mean_kl"] = mean_kl
            metrics[f"{prefix}logps_diff"] = float(loss_metrics["logps_diff"])
            metrics[f"{prefix}beta"] = float(self.beta)
            metrics[f"{prefix}rewards/margins"] = float(loss_metrics.get("rewards/margins", 0.0))
            metrics[f"{prefix}logps/chosen"] = float(
                model_output["chosen_logps"].detach().mean().item()
            )
            metrics[f"{prefix}logps/rejected"] = float(
                model_output["rejected_logps"].detach().mean().item()
            )

            if train_eval == "train":
                self._last_train_kl = mean_kl
            else:
                self._last_eval_kl = mean_kl
                self._last_eval_loss = float(loss.detach().item())

            return loss, metrics

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            from contextlib import nullcontext

            try:
                from torch import autocast
            except ImportError:  # pragma: no cover
                autocast = None  # type: ignore

            ctx = nullcontext()
            if (
                autocast is not None
                and getattr(self, "_peft_has_been_casted_to_bf16", False)
                and hasattr(self, "accelerator")
            ):
                ctx = autocast(self.accelerator.device.type)

            with ctx:
                try:
                    loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")
                except TypeError:
                    loss, metrics = self.get_batch_loss_metrics(model, inputs)

            if hasattr(self, "accelerator"):
                loss = loss.to(self.args.device)
            if hasattr(self, "store_metrics"):
                self.store_metrics(metrics, train_eval="train")

            # 逐步上报（梯度同步步）→ MetricSink
            should_log = True
            accel = getattr(self, "accelerator", None)
            if accel is not None and hasattr(accel, "sync_gradients"):
                should_log = bool(accel.sync_gradients)
            if should_log:
                step = int(getattr(getattr(self, "state", None), "global_step", 0) or 0)
                self._emit_metrics(
                    {
                        "train/loss": float(loss.detach().item()),
                        "train/mean_kl": float(metrics.get("mean_kl", self._last_train_kl)),
                        "train/beta": float(self.beta),
                    },
                    step=step,
                )

            if return_outputs:
                return loss, metrics
            return loss

        def evaluation_loop(self, *args: Any, **kwargs: Any):
            output = super().evaluation_loop(*args, **kwargs)
            # 验证结束后用验证 KL 更新动态 β
            eval_kl = self._aggregate_stored_eval_kl()
            if eval_kl is None:
                eval_kl = self._last_eval_kl
            eval_loss = self._last_eval_loss
            step = int(getattr(getattr(self, "state", None), "global_step", 0) or 0)
            self._emit_metrics(
                {"val/loss": float(eval_loss), "val/mean_kl": float(eval_kl)},
                step=step,
            )

            if self.beta_controller is not None and self.beta_mode == "dynamic":
                new_beta = self.beta_controller.step(float(eval_kl))
                self.beta = float(new_beta)
                if hasattr(self, "args") and hasattr(self.args, "beta"):
                    self.args.beta = self.beta
                logger.info(
                    "动态 β 更新: kl=%.4f target=%.4f → beta=%.6f",
                    float(eval_kl),
                    self.beta_controller.target_kl,
                    self.beta,
                )
                self._emit_metrics(
                    {"train/beta": float(self.beta), "val/mean_kl": float(eval_kl)},
                    step=step,
                )
            return output

        def _aggregate_stored_eval_kl(self) -> float | None:
            stored = getattr(self, "_stored_metrics", None)
            if not stored:
                return None
            values = stored.get("eval", {}).get("eval_mean_kl") or stored.get("eval", {}).get("mean_kl")
            if not values:
                return None
            return float(sum(values) / max(len(values), 1))

        def _save_checkpoint(self, model, trial, metrics=None):
            # 不同 transformers 版本签名不一致
            try:
                result = super()._save_checkpoint(model, trial, metrics=metrics)
            except TypeError:
                try:
                    result = super()._save_checkpoint(model, trial)
                except TypeError:
                    result = super()._save_checkpoint(model, trial, metrics)

            if self.beta_controller is not None and self.args.should_save:
                ckpt_dir = Path(self.state.best_model_checkpoint or self.args.output_dir)
                # HF 刚写入的是 checkpoint-N
                if self.state.global_step:
                    step_dir = Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}"
                    if step_dir.is_dir():
                        ckpt_dir = step_dir
                self._save_beta_state(ckpt_dir)
            return result

        def _save_beta_state(self, directory: Path) -> None:
            if self.beta_controller is None:
                return
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / BETA_STATE_FILENAME
            path.write_text(json.dumps(self.beta_controller.state_dict(), indent=2), encoding="utf-8")

        def _load_beta_state(self, checkpoint: str | Path) -> None:
            if self.beta_controller is None:
                return
            path = Path(checkpoint) / BETA_STATE_FILENAME
            if not path.is_file():
                return
            state = json.loads(path.read_text(encoding="utf-8"))
            self.beta_controller.load_state_dict(state)
            self.beta = float(self.beta_controller.beta)
            if hasattr(self, "args") and hasattr(self.args, "beta"):
                self.args.beta = self.beta
            logger.info("已从 checkpoint 恢复动态 β=%.6f", self.beta)

        def train(self, *args: Any, **kwargs: Any):
            resume = kwargs.get("resume_from_checkpoint")
            if resume:
                self._load_beta_state(resume)
            return super().train(*args, **kwargs)

    _CustomDPOTrainer.__name__ = "CustomDPOTrainer"
    _CustomDPOTrainer.__qualname__ = "CustomDPOTrainer"
    return _CustomDPOTrainer


@register(ALGORITHM_REGISTRY, "dpo")
class DPOStageTrainer(BaseTrainer):
    """
    偏好对齐策略。

    - Loss / IPO：自研（``dpo_losses``）
    - 动态 β：``DynamicBetaController``
    - 训练循环 / LoRA / AMP / DDP / 续训 / WandB：TRL ``DPOTrainer``（底层 Accelerate）
    """

    name = "dpo"

    def run_algorithm(self, ctx: TrainContext) -> TrainResult:
        from trl import DPOConfig

        tcfg = ctx.config.training
        runtime = ctx.runtime
        loss_cfg = _build_loss_cfg(tcfg)
        out_dir = ctx.checkpoint_dir
        beta_controller = _build_beta_controller(loss_cfg)

        model = self.load_adapter_resume(ctx.model, ctx.resume_from)
        # LoRA 场景让 TRL 用 disable_adapter 作 ref；全参则由 TRL 克隆 ref
        ref_model = None

        max_steps = tcfg.get("max_steps", None)
        eval_ds = ctx.eval_dataset
        eval_strategy = str(tcfg.get("eval_strategy", "epoch" if eval_ds is not None else "no"))
        if eval_ds is None:
            eval_strategy = "no"

        dpo_kwargs: dict[str, Any] = {
            "output_dir": out_dir,
            "num_train_epochs": float(tcfg.get("num_train_epochs", 1)),
            "per_device_train_batch_size": int(tcfg.get("per_device_train_batch_size", 1)),
            "per_device_eval_batch_size": int(
                tcfg.get("per_device_eval_batch_size", tcfg.get("per_device_train_batch_size", 1))
            ),
            "logging_steps": int(tcfg.get("logging_steps", 10)),
            "eval_strategy": eval_strategy,
            "save_strategy": str(tcfg.get("save_strategy", "epoch")),
            "max_length": int(tcfg.get("max_length", 1024)),
            "max_prompt_length": int(tcfg.get("max_prompt_length", 512)),
            "beta": float(loss_cfg.beta),
            # TRL 内部仍要合法 loss_type；真实损失由 CustomDPOTrainer 接管
            "loss_type": "sigmoid",
            "remove_unused_columns": False,
            **ctx.hf_args,
        }
        if max_steps not in (None, "null"):
            dpo_kwargs["max_steps"] = int(max_steps)

        dpo_kwargs = _filter_kwargs(DPOConfig, dpo_kwargs)
        args = DPOConfig(**dpo_kwargs)

        CustomDPOTrainerCls = build_custom_dpo_trainer_cls()
        trainer_init = {
            "model": model,
            "ref_model": ref_model,
            "args": args,
            "train_dataset": ctx.train_dataset,
            "eval_dataset": eval_ds,
            # TRL 0.15+ 用 processing_class；勿再传 tokenizer（0.29 会 TypeError）
            "processing_class": ctx.tokenizer,
            "callbacks": [self.make_hf_metrics_callback()],
            "custom_loss_type": loss_cfg.loss_type,
            "beta_mode": loss_cfg.resolved_beta_mode(),
            "margin": loss_cfg.margin,
            "beta_controller": beta_controller,
            "metrics_fn": self.track_metrics,
        }
        trainer_init = _filter_kwargs(CustomDPOTrainerCls.__init__, trainer_init)
        # 自定义 kw 可能被 filter 掉（父类签名无 **kwargs 时不会），强制写回
        trainer_init["custom_loss_type"] = loss_cfg.loss_type
        trainer_init["beta_mode"] = loss_cfg.resolved_beta_mode()
        trainer_init["margin"] = loss_cfg.margin
        trainer_init["beta_controller"] = beta_controller
        trainer_init["metrics_fn"] = self.track_metrics
        if "callbacks" not in trainer_init:
            # 父类 __init__ 若接受 callbacks，上面 filter 会留下；否则挂到 args 侧无效时再尝试
            pass

        report_to = dpo_kwargs.get("report_to", getattr(args, "report_to", "none"))
        logger.info(
            "DPO(TRL/Accelerate) 启动 loss_type=%s beta_mode=%s beta=%.4f margin=%.2f report_to=%s",
            loss_cfg.loss_type,
            loss_cfg.resolved_beta_mode(),
            loss_cfg.beta,
            loss_cfg.margin,
            report_to,
        )
        trainer = CustomDPOTrainerCls(**trainer_init)
        if ctx.resume_from and hasattr(trainer, "_load_beta_state"):
            trainer._load_beta_state(ctx.resume_from)

        train_out = trainer.train(resume_from_checkpoint=ctx.resume_from)
        metrics = dict(train_out.metrics) if train_out and getattr(train_out, "metrics", None) else {}
        if eval_ds is not None and eval_strategy != "no":
            try:
                metrics.update(trainer.evaluate())
            except Exception as exc:
                logger.warning("DPO evaluate 跳过: %s", exc)

        metrics["dpo/configured_beta"] = float(getattr(trainer, "beta", loss_cfg.beta))
        metrics["dpo/dynamic_beta"] = 1.0 if loss_cfg.resolved_beta_mode() == "dynamic" else 0.0
        metrics["dpo/loss_type"] = 0.0 if loss_cfg.loss_type == "dpo" else 1.0
        if ctx.init_from:
            metrics["dpo/init_from"] = 1.0
            logger.info("DPO 自上游冷启动: %s", ctx.init_from)
        self.save(getattr(trainer, "model", model), ctx.tokenizer, out_dir)
        if beta_controller is not None:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            (Path(out_dir) / BETA_STATE_FILENAME).write_text(
                json.dumps(beta_controller.state_dict(), indent=2), encoding="utf-8"
            )

        return TrainResult(
            metrics=metrics,
            checkpoint_dir=out_dir,
            extra={
                "loss_cfg": loss_cfg,
                "impl": "trl",
                "beta_mode": loss_cfg.resolved_beta_mode(),
            },
        )
