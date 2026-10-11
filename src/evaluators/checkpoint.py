"""评测用 Checkpoint 解析与模型加载（生产实现，可注入依赖做 CPU 单测）。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

from src.core.base import STAGE_CHECKPOINT_SUBDIRS, BaseTrainer

logger = logging.getLogger("posttrainlab.eval.checkpoint")

LoadModelFn = Callable[..., Tuple[Any, Any]]
LoadAdapterFn = Callable[[Any, str], Any]


class CheckpointError(ValueError):
    """非法或不存在的 checkpoint。"""


def detect_checkpoint_kind(path: Path) -> str:
    """
    识别目录类型。

    - ``lora``：含 adapter_config.json
    - ``full``：含 HF config.json（全量微调权重）
    - 否则抛 :class:`CheckpointError`
    """
    if not path.exists():
        raise CheckpointError(f"Checkpoint 不存在: {path}")
    if not path.is_dir():
        raise CheckpointError(f"Checkpoint 必须是目录: {path}")
    if (path / "adapter_config.json").is_file():
        return "lora"
    if (path / "config.json").is_file():
        return "full"
    raise CheckpointError(
        f"无法识别的 checkpoint（缺 adapter_config.json / config.json）: {path}"
    )


def resolve_checkpoint(
    config: Any,
    checkpoint: str | Path | None = None,
    *,
    output_dir: str | None = None,
) -> Path:
    """
    显式选择待评测 checkpoint。

    优先级：
    1. 函数参数 ``checkpoint``
    2. ``config.eval.checkpoint``
    3. 阶段别名（sft/dpo/grpo/ppo）→ ``{output_dir}/checkpoints/<stage>`` 最新
    4. 目录路径；若含 ``checkpoint-*`` 则取最新
    """
    from src.evaluators.common import eval_section

    raw = checkpoint
    if raw in (None, "", "null", "none"):
        ecfg = eval_section(config)
        raw = ecfg.get("checkpoint")

    if raw in (None, "", "null", "none"):
        raise CheckpointError(
            "未指定 checkpoint。请传 evaluate(..., checkpoint=...) "
            "或设置 eval.checkpoint（路径 / sft|dpo|grpo|ppo / latest）"
        )

    key = str(raw).strip()
    root = Path(output_dir or _cfg_get(config, "output_dir", "outputs"))

    if key.lower() in ("true", "auto", "latest"):
        # 扫描常见阶段目录，取全局最新一步（按 mtime）
        candidates: list[Path] = []
        for sub in STAGE_CHECKPOINT_SUBDIRS.values():
            found = BaseTrainer.latest_checkpoint(str(root / sub))
            if found:
                candidates.append(Path(found))
        if not candidates:
            raise CheckpointError(f"output_dir={root} 下未找到任何 checkpoint-*")
        return max(candidates, key=lambda p: p.stat().st_mtime)

    if key.lower() in STAGE_CHECKPOINT_SUBDIRS:
        stage_root = root / STAGE_CHECKPOINT_SUBDIRS[key.lower()]
        found = BaseTrainer.latest_checkpoint(str(stage_root))
        if not found:
            raise CheckpointError(f"阶段 '{key}' 无可用 checkpoint: {stage_root}")
        return Path(found)

    path = Path(key)
    if not path.is_absolute():
        # 相对路径：先按 cwd，再不存在则相对 output_dir
        if not path.exists() and (root / path).exists():
            path = root / path

    if path.is_dir():
        # 若本身是 stage 根目录，解析到最新 checkpoint-*
        nested = BaseTrainer.latest_checkpoint(str(path))
        if nested and Path(nested) != path:
            # latest_checkpoint 在目录本身含 trainer_state/adapter 时会返回自身
            if (path / "adapter_config.json").is_file() or (path / "config.json").is_file():
                return path
            return Path(nested)
        if (path / "adapter_config.json").is_file() or (path / "config.json").is_file():
            return path
        nested2 = BaseTrainer.latest_checkpoint(str(path))
        if nested2:
            return Path(nested2)
        raise CheckpointError(f"目录内无有效 checkpoint: {path}")

    if not path.exists():
        raise CheckpointError(f"Checkpoint 路径不存在: {path}")
    raise CheckpointError(f"Checkpoint 必须是目录: {path}")


def load_checkpoint_for_eval(
    config: Any,
    checkpoint: str | Path,
    *,
    load_model_fn: Optional[LoadModelFn] = None,
    load_adapter_fn: Optional[LoadAdapterFn] = None,
) -> Tuple[Any, Any, str]:
    """
    按 checkpoint 类型加载模型与 tokenizer（生产路径）。

    - **full**：从 checkpoint 目录加载权重与 tokenizer
    - **lora**：从 ``config.model.name`` 加载基座，再挂 adapter；tokenizer 优先 checkpoint 内

    返回 ``(model, tokenizer, kind)``。可通过 ``load_model_fn`` / ``load_adapter_fn`` 注入做单测。
    """
    path = Path(checkpoint)
    kind = detect_checkpoint_kind(path)

    load_model = load_model_fn or _default_load_model
    load_adapter = load_adapter_fn or _default_load_adapter

    use_gpu = bool(_cfg_get(config, "use_gpu", True))
    model_cfg = _section(config, "model")
    torch_dtype = model_cfg.get("torch_dtype")
    trust = bool(model_cfg.get("trust_remote_code", True))
    chat_cfg = _cfg_get(config, "chat_template", None)

    if kind == "full":
        logger.info("评测加载 Full checkpoint: %s", path)
        model, tokenizer = load_model(
            str(path),
            use_gpu=use_gpu,
            torch_dtype=torch_dtype,
            trust_remote_code=trust,
            chat_cfg=chat_cfg,
        )
        return model, tokenizer, kind

    # LoRA：基座 + adapter
    base_name = str(model_cfg.get("name") or model_cfg.get("model_name") or "")
    if not base_name:
        raise CheckpointError("LoRA 评测需要 config.model.name 指定基座模型")
    logger.info("评测加载 LoRA：base=%s adapter=%s", base_name, path)
    model, tokenizer = load_model(
        base_name,
        use_gpu=use_gpu,
        torch_dtype=torch_dtype,
        trust_remote_code=trust,
        chat_cfg=chat_cfg,
    )
    model = load_adapter(model, str(path))
    # adapter 目录常带 tokenizer 文件，优先使用
    tok_from_ckpt = _try_load_tokenizer(path, trust_remote_code=trust, chat_cfg=chat_cfg)
    if tok_from_ckpt is not None:
        tokenizer = tok_from_ckpt
    return model, tokenizer, kind


def _default_load_model(
    name_or_path: str,
    *,
    use_gpu: bool = True,
    torch_dtype: Any = None,
    trust_remote_code: bool = True,
    chat_cfg: Any = None,
) -> tuple[Any, Any]:
    from src.models.loader import load_model_and_tokenizer

    return load_model_and_tokenizer(
        name_or_path,
        use_gpu=use_gpu,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        chat_cfg=chat_cfg,
    )


def _default_load_adapter(model: Any, adapter_path: str) -> Any:
    """推理态挂载 LoRA（is_trainable=False）。"""
    from peft import PeftModel

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    try:
        loaded = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
    except TypeError:
        loaded = PeftModel.from_pretrained(base, adapter_path)
    loaded.eval()
    return loaded


def _try_load_tokenizer(path: Path, *, trust_remote_code: bool, chat_cfg: Any):
    if not (path / "tokenizer_config.json").is_file() and not (path / "tokenizer.json").is_file():
        return None
    try:
        from transformers import AutoTokenizer

        from src.data.chat_templates import apply_chat_template_config

        tok = AutoTokenizer.from_pretrained(str(path), trust_remote_code=trust_remote_code)
        apply_chat_template_config(tok, chat_cfg)
        if not tok.pad_token:
            tok.pad_token = tok.eos_token
        return tok
    except Exception as exc:
        logger.warning("未能从 checkpoint 加载 tokenizer，沿用基座: %s", exc)
        return None


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        try:
            val = config.get(key, default)
            return default if val is None else val
        except Exception:
            pass
    return getattr(config, key, default)


def _section(config: Any, name: str) -> dict[str, Any]:
    raw = _cfg_get(config, name, {})
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
