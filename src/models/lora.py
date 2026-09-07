from __future__ import annotations

from typing import Any

from peft import LoraConfig, PeftModel, get_peft_model


def apply_lora(model: Any, lora_cfg: Any):
    """Wrap a causal LM with LoRA adapters from Hydra config."""
    config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        target_modules=list(lora_cfg.target_modules),
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
    )
    return get_peft_model(model, config)


def merge_lora(model: PeftModel):
    """Merge LoRA weights into the base model for export / eval."""
    return model.merge_and_unload()
