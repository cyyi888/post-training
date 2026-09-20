from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.chat_templates import apply_chat_template_config


def load_model_and_tokenizer(
    model_name: str,
    use_gpu: bool = True,
    torch_dtype: str | torch.dtype | None = None,
    trust_remote_code: bool = True,
    chat_cfg: Any | None = None,
    **kwargs: Any,
):
    dtype = None
    if isinstance(torch_dtype, str):
        dtype = getattr(torch, torch_dtype, None)
    elif torch_dtype is not None:
        dtype = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
        **kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )

    if use_gpu and torch.cuda.is_available():
        model.to("cuda")
    elif use_gpu:
        print("Warning: CUDA unavailable, using CPU.")

    apply_chat_template_config(tokenizer, chat_cfg)

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
