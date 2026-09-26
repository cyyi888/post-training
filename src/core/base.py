"""统一训练接口：算法无关的工程能力在基类，子类只实现算法。"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrainResult:
    """所有训练策略的统一返回值。"""

    metrics: dict[str, float] = field(default_factory=dict)
    checkpoint_dir: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeOptions:
    """与具体算法无关的优化 / 显存设置。"""

    seed: int = 0
    learning_rate: float = 1.0e-5
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    mixed_precision: str = "no"  # no | fp16 | bf16
    lr_scheduler_type: str = "cosine"  # cosine | linear | constant
    warmup_ratio: float = 0.0
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0


@dataclass
class TrainContext:
    """一次训练的工程上下文，由基类组装后交给算法。"""

    config: Any
    dataset: Any
    train_dataset: Any
    eval_dataset: Any
    model: Any
    tokenizer: Any
    checkpoint_dir: str
    resume_from: str | None = None
    hf_args: dict[str, Any] = field(default_factory=dict)
    runtime: RuntimeOptions = field(default_factory=RuntimeOptions)


class BaseTrainer(ABC):
    """
    训练策略基类。

    对外仍是 ``train(config, dataset)``。基类负责与算法无关的工程能力：
    日志、实验追踪、断点续训、分布式，以及梯度累积、混合精度、
    梯度检查点、随机种子和学习率调度。子类只实现 :meth:`run_algorithm`。
    """

    name: str = "base"

    def __init__(self) -> None:
        self.logger = logging.getLogger("posttrainlab.trainer")
        self.global_rank = 0
        self.local_rank = 0
        self.world_size = 1
        self.is_main_process = True
        self._dist_owned = False
        self._tracking = False

    def train(self, config: Any, dataset: Any) -> TrainResult:
        """模板方法：工程准备 → 算法 → 记录与收尾。"""
        self.prepare_engineering(config)
        try:
            ctx = self.build_context(config, dataset)
            self.logger.info(
                "开始 %s | rank %s/%s | resume=%s | accum=%s | amp=%s | scheduler=%s",
                self.name,
                self.global_rank,
                self.world_size,
                ctx.resume_from,
                ctx.runtime.gradient_accumulation_steps,
                ctx.runtime.mixed_precision,
                ctx.runtime.lr_scheduler_type,
            )
            result = self.run_algorithm(ctx)
            if self.is_main_process:
                self.track_metrics({f"final/{k}": v for k, v in result.metrics.items()})
                self.logger.info(
                    "训练结束 checkpoint=%s metrics=%s",
                    result.checkpoint_dir,
                    result.metrics,
                )
            return result
        finally:
            self.finish_engineering()

    @abstractmethod
    def run_algorithm(self, ctx: TrainContext) -> TrainResult:
        """子类只实现算法本身：损失、采样或调用 TRL Trainer。"""

    # ----- 工程：日志 / 追踪 / 分布式 / 续训 -----

    def prepare_engineering(self, config: Any) -> None:
        """初始化日志、种子、分布式信息，以及主进程上的实验追踪。"""
        from src.utils.logging_utils import setup_logging

        self.detect_distributed(config)
        if self.is_main_process or config.get("logging", {}).get("all_ranks", False):
            self.logger = setup_logging(config)
        else:
            self.logger = logging.getLogger(f"posttrainlab.rank{self.global_rank}")
            self.logger.setLevel(logging.WARNING)

        try:
            from src.utils.seed import same_seeds

            same_seeds(int(config.get("seed", 0)))
        except Exception as exc:
            self.logger.warning("未能固定随机种子: %s", exc)

        self._tracking = False
        if self.is_main_process:
            try:
                from src.utils.wandb_utils import init_wandb, resolve_report_to

                try:
                    from omegaconf import OmegaConf

                    OmegaConf.set_struct(config, False)
                except Exception:
                    pass
                if hasattr(config, "report_to"):
                    config.report_to = resolve_report_to(config)
                init_wandb(config)
                self._tracking = True
                self._log_bound_data_version(config)
            except Exception as exc:
                self.logger.warning("实验追踪未启用: %s", exc)
        else:
            os.environ.setdefault("WANDB_DISABLED", "true")

        self.logger.info(
            "工程就绪 | algo=%s distributed=%s world=%s",
            self.name,
            self._distributed_mode(config),
            self.world_size,
        )

    def finish_engineering(self) -> None:
        if self._tracking and self.is_main_process:
            try:
                from src.utils.wandb_utils import finish_wandb

                finish_wandb()
            except Exception as exc:
                self.logger.warning("结束实验追踪失败: %s", exc)
        if self._dist_owned:
            try:
                import torch.distributed as dist

                if dist.is_initialized():
                    dist.destroy_process_group()
            except Exception as exc:
                self.logger.warning("关闭进程组失败: %s", exc)
            self._dist_owned = False

    def track_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """主进程把标量写到日志和 W&B。"""
        if not self.is_main_process or not metrics:
            return
        self.logger.info("metrics %s", metrics)
        if not self._tracking:
            return
        try:
            from src.utils.wandb_utils import log_metrics

            log_metrics(metrics, step=step)
        except Exception as exc:
            self.logger.warning("上报指标失败: %s", exc)

    def detect_distributed(self, config: Any) -> None:
        """读取 torchrun / accelerate 注入的环境变量。``distributed: none`` 时强制单进程。"""
        mode = self._distributed_mode(config)
        if mode == "none":
            self.world_size = 1
            self.global_rank = 0
            self.local_rank = 0
        else:
            self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
            self.global_rank = int(os.environ.get("RANK", "0"))
            self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.is_main_process = self.global_rank == 0

    def ensure_process_group(self) -> None:
        """自研训练循环需要时再初始化进程组。HF Trainer 会自己初始化，不要重复调用。"""
        if self.world_size <= 1:
            return
        import torch

        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
        import torch.distributed as dist

        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
            self._dist_owned = True

    def prepare_native_model(self, model: Any, resume_from: str | None) -> Any:
        """自研循环：可选恢复 LoRA 权重，多卡时包一层 DDP。"""
        model = self.load_adapter_resume(model, resume_from)
        if self.world_size <= 1:
            return model
        self.ensure_process_group()
        import torch
        from torch.nn.parallel import DistributedDataParallel as DDP

        device_ids = [self.local_rank] if torch.cuda.is_available() else None
        return DDP(model, device_ids=device_ids)

    def hf_training_kwargs(self, config: Any) -> dict[str, Any]:
        """传给 TRL / HF TrainingArguments 的通用字段（含优化与精度）。"""
        from src.utils.wandb_utils import default_run_name, resolve_report_to

        runtime = getattr(self, "runtime", None) or self.resolve_runtime(config)
        kwargs: dict[str, Any] = {
            "report_to": resolve_report_to(config),
            "run_name": default_run_name(config),
            "logging_dir": f"{config.output_dir}/logs/{config.training.algorithm}",
            "learning_rate": runtime.learning_rate,
            "gradient_accumulation_steps": runtime.gradient_accumulation_steps,
            "gradient_checkpointing": runtime.gradient_checkpointing,
            "lr_scheduler_type": runtime.lr_scheduler_type,
            "warmup_ratio": runtime.warmup_ratio,
            "weight_decay": runtime.weight_decay,
            "max_grad_norm": runtime.max_grad_norm,
            "seed": runtime.seed,
            "fp16": runtime.mixed_precision == "fp16",
            "bf16": runtime.mixed_precision == "bf16",
        }
        if self.world_size > 1:
            kwargs["local_rank"] = self.local_rank
            kwargs["ddp_find_unused_parameters"] = bool(
                self._engineering(config).get("ddp_find_unused_parameters", False)
            )
        return kwargs

    def resolve_runtime(self, config: Any) -> RuntimeOptions:
        """合并 engineering 与 training 中的通用优化项。engineering 优先，空值回退到 training。"""
        requested = str(self._pick(config, "mixed_precision", "auto"))
        return RuntimeOptions(
            seed=int(config.get("seed", 0) if hasattr(config, "get") else 0),
            learning_rate=float(self._pick(config, "learning_rate", 1.0e-5)),
            gradient_accumulation_steps=max(1, int(self._pick(config, "gradient_accumulation_steps", 1))),
            gradient_checkpointing=self._as_bool(self._pick(config, "gradient_checkpointing", False)),
            mixed_precision=self.resolve_mixed_precision(requested),
            lr_scheduler_type=str(self._pick(config, "lr_scheduler_type", "cosine")),
            warmup_ratio=float(self._pick(config, "warmup_ratio", 0.0)),
            weight_decay=float(self._pick(config, "weight_decay", 0.0)),
            max_grad_norm=float(self._pick(config, "max_grad_norm", 1.0)),
        )

    @staticmethod
    def resolve_mixed_precision(requested: str) -> str:
        """``auto``：有 CUDA 且支持 bf16 则用 bf16，否则 fp16；无 CUDA 则 fp32。"""
        req = str(requested).lower()
        if req in ("no", "fp32", "false", "none"):
            return "no"
        if req in ("fp16", "bf16"):
            return req
        try:
            import torch

            if not torch.cuda.is_available():
                return "no"
            supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            return "bf16" if supported else "fp16"
        except Exception:
            return "no"

    def apply_gradient_checkpointing(self, model: Any, enabled: bool) -> Any:
        if not enabled:
            return model
        raw = model.module if hasattr(model, "module") else model
        if hasattr(raw, "gradient_checkpointing_enable"):
            raw.gradient_checkpointing_enable()
            if hasattr(raw, "enable_input_require_grads"):
                raw.enable_input_require_grads()
            self.logger.info("已开启梯度检查点")
        else:
            self.logger.warning("当前模型不支持 gradient checkpointing")
        return model

    def build_optimizer_and_scheduler(self, model: Any, num_update_steps: int):
        """自研训练循环使用的 AdamW + warmup 调度。返回 (optimizer, scheduler, grad_scaler|None)。"""
        import torch
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import LambdaLR

        runtime = self.runtime
        optimizer = AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=runtime.learning_rate,
            weight_decay=runtime.weight_decay,
        )
        warmup = int(num_update_steps * runtime.warmup_ratio)
        kind = runtime.lr_scheduler_type

        def lr_lambda(step: int) -> float:
            if warmup > 0 and step < warmup:
                return float(step + 1) / float(max(warmup, 1))
            if kind == "constant":
                return 1.0
            progress = (step - warmup) / float(max(1, num_update_steps - warmup))
            progress = min(max(progress, 0.0), 1.0)
            if kind == "linear":
                return max(0.0, 1.0 - progress)
            import math

            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda)
        scaler = None
        if runtime.mixed_precision == "fp16" and torch.cuda.is_available():
            try:
                scaler = torch.amp.GradScaler("cuda")
            except (TypeError, AttributeError):
                scaler = torch.cuda.amp.GradScaler()
        return optimizer, scheduler, scaler

    @staticmethod
    def estimate_update_steps(
        num_examples: int,
        batch_size: int,
        epochs: int,
        grad_accum: int,
    ) -> int:
        import math

        micros = math.ceil(num_examples / float(max(1, batch_size))) * max(1, epochs)
        return max(1, math.ceil(micros / float(max(1, grad_accum))))

    def build_context(self, config: Any, dataset: Any) -> TrainContext:
        self.runtime = self.resolve_runtime(config)
        train_ds, eval_ds = self.resolve_splits(dataset)
        model, tokenizer = self.setup_model(config)
        model = self.apply_gradient_checkpointing(model, self.runtime.gradient_checkpointing)
        out_dir = self.checkpoint_dir(config)
        resume_from = self.resolve_resume(config, out_dir)
        if resume_from:
            self.logger.info("将从 checkpoint 续训: %s", resume_from)
        self.logger.info(
            "优化设置 lr=%s accum=%s amp=%s ckpt=%s scheduler=%s warmup=%s seed=%s",
            self.runtime.learning_rate,
            self.runtime.gradient_accumulation_steps,
            self.runtime.mixed_precision,
            self.runtime.gradient_checkpointing,
            self.runtime.lr_scheduler_type,
            self.runtime.warmup_ratio,
            self.runtime.seed,
        )
        return TrainContext(
            config=config,
            dataset=dataset,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            model=model,
            tokenizer=tokenizer,
            checkpoint_dir=out_dir,
            resume_from=resume_from,
            hf_args=self._safe_hf_kwargs(config),
            runtime=self.runtime,
        )

    def resolve_resume(self, config: Any, checkpoint_dir: str) -> str | None:
        """
        ``engineering.resume_from_checkpoint``：
        null 不续训；true / auto / latest 找最新 checkpoint-*；否则视为路径。
        """
        flag = self._engineering(config).get("resume_from_checkpoint", None)
        if flag in (None, False, "", "null", "none", "false"):
            return None
        if flag is True or str(flag).lower() in ("true", "auto", "latest"):
            return self.latest_checkpoint(checkpoint_dir)
        path = Path(str(flag))
        if not path.exists():
            self.logger.warning("指定的 checkpoint 不存在: %s", path)
            return None
        return str(path)

    @staticmethod
    def latest_checkpoint(directory: str) -> str | None:
        root = Path(directory)
        if not root.is_dir():
            return None
        candidates = [p for p in root.glob("checkpoint-*") if p.is_dir()]
        if candidates:
            def _step(path: Path) -> int:
                try:
                    return int(path.name.split("-")[-1])
                except ValueError:
                    return -1

            return str(max(candidates, key=_step))
        if (root / "trainer_state.json").is_file() or (root / "adapter_config.json").is_file():
            return str(root)
        return None

    def load_adapter_resume(self, model: Any, resume_from: str | None) -> Any:
        """若目录里是 LoRA adapter，则加载到当前模型上。"""
        if not resume_from:
            return model
        path = Path(resume_from)
        if not (path / "adapter_config.json").is_file():
            return model
        from peft import PeftModel

        self.logger.info("加载 LoRA checkpoint: %s", path)
        return PeftModel.from_pretrained(model, str(path))

    def save(self, model: Any, tokenizer: Any, path: str) -> None:
        """仅主进程保存；若模型被 DDP 包裹则保存内部模块。"""
        if not getattr(self, "is_main_process", True):
            return
        Path(path).mkdir(parents=True, exist_ok=True)
        raw = model.module if hasattr(model, "module") else model
        raw.save_pretrained(path)
        tokenizer.save_pretrained(path)

    @staticmethod
    def resolve_splits(dataset: Any) -> tuple[Any, Any]:
        """从 DatasetDict / dict / 单 split 解析 (train, eval)。"""
        if dataset is None:
            raise ValueError("dataset 不能为空")
        if hasattr(dataset, "keys"):
            keys = set(dataset.keys())
            if "train" in keys:
                train = dataset["train"]
                eval_ds = dataset["eval"] if "eval" in keys else dataset.get("test")
                return train, eval_ds
        return dataset, None

    @staticmethod
    def setup_model(config: Any) -> tuple[Any, Any]:
        """按配置加载模型 + tokenizer，并按需挂 LoRA。"""
        from src.models.loader import load_model_and_tokenizer
        from src.models.lora import apply_lora

        model, tokenizer = load_model_and_tokenizer(
            config.model.name,
            use_gpu=config.get("use_gpu", True),
            torch_dtype=config.model.get("torch_dtype"),
            trust_remote_code=config.model.get("trust_remote_code", True),
            chat_cfg=config.get("chat_template"),
        )
        if config.model.get("use_lora", False):
            model = apply_lora(model, config.model.lora)
            # peft 模型：打印可训练比例便于确认 Full / LoRA
            if hasattr(model, "print_trainable_parameters"):
                model.print_trainable_parameters()
        else:
            # Full：确保全部参数可训练
            for p in model.parameters():
                p.requires_grad = True
        return model, tokenizer

    @staticmethod
    def checkpoint_dir(config: Any) -> str:
        return f"{config.output_dir}/{config.training.output_subdir}"

    def _log_bound_data_version(self, config: Any) -> None:
        version = config.get("data_version") if hasattr(config, "get") else None
        if not isinstance(version, dict):
            return
        self.logger.info(
            "绑定数据版本 version=%s hash=%s",
            version.get("version"),
            str(version.get("hash", ""))[:12],
        )
        self.track_metrics(
            {
                "data/version": version.get("version", ""),
                "data/hash": str(version.get("hash", ""))[:12],
                "data/num_samples": version.get("num_samples", 0),
            }
        )

    def _safe_hf_kwargs(self, config: Any) -> dict[str, Any]:
        try:
            return self.hf_training_kwargs(config)
        except Exception as exc:
            self.logger.warning("未能组装 HF 日志参数: %s", exc)
            return {}

    @staticmethod
    def _engineering(config: Any) -> dict[str, Any]:
        eng = config.get("engineering") if hasattr(config, "get") else None
        if eng is None:
            return {}
        if isinstance(eng, dict):
            return dict(eng)
        if hasattr(eng, "items"):
            try:
                return dict(eng.items())
            except Exception:
                pass
        data = getattr(eng, "__dict__", None)
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if not str(k).startswith("_")}
        return {}

    def _distributed_mode(self, config: Any) -> str:
        return str(self._engineering(config).get("distributed", "auto"))

    def _pick(self, config: Any, key: str, default: Any) -> Any:
        """engineering 中的非空值优先，否则用 training 同名字段。"""
        eng_val = self._engineering(config).get(key, None)
        if not self._is_blank(eng_val):
            return eng_val
        training = self._section(config, "training")
        train_val = training.get(key, None)
        if not self._is_blank(train_val):
            return train_val
        return default

    @staticmethod
    def _is_blank(value: Any) -> bool:
        return value is None or value == "" or str(value).lower() in ("null", "none")

    def _section(self, config: Any, name: str) -> dict[str, Any]:
        if name == "engineering":
            return self._engineering(config)
        raw = config.get(name) if hasattr(config, "get") else None
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return dict(raw)
        if hasattr(raw, "items"):
            try:
                return dict(raw.items())
            except Exception:
                pass
        data = getattr(raw, "__dict__", None)
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if not str(k).startswith("_")}
        return {}

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).lower() in ("1", "true", "yes")


class BaseEvaluator(ABC):
    """统一评测接口。"""

    def __init__(self, cfg: Any, model: Any, tokenizer: Any):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer

    @abstractmethod
    def evaluate(self, dataset: Any) -> dict[str, float]:
        """Return metric name → value."""
