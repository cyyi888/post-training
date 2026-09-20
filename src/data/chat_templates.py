"""Chat Template 统一管理：按基座模型集中配置，训练/推理共用。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("posttrainlab.data.chat_templates")

# 内置兜底 Jinja（无 tokenizer.chat_template 且无配置覆盖时使用）
DEFAULT_CHAT_JINJA = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "System: {{ message['content'] }}\n"
    "{% elif message['role'] == 'user' %}"
    "User: {{ message['content'] }}\n"
    "{% elif message['role'] == 'assistant' %}"
    "Assistant: {% generation %}{{ message['content'] }}{{ eos_token }}{% endgeneration %}\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}Assistant: {% endif %}"
)


def apply_chat_template_config(tokenizer: Any, chat_cfg: Any | None = None) -> Any:
    """
    根据 Hydra ``chat_template`` 配置设置 tokenizer.chat_template。

    chat_cfg.source:
      - tokenizer: 优先保留模型自带；缺失则用 inline/jinja_file/DEFAULT
      - inline: 强制使用 chat_cfg.jinja
      - jinja_file: 从文件加载
    """
    if chat_cfg is None:
        if not getattr(tokenizer, "chat_template", None):
            tokenizer.chat_template = DEFAULT_CHAT_JINJA
            logger.info("使用内置默认 chat template")
        return tokenizer

    source = str(chat_cfg.get("source", "tokenizer"))
    jinja = chat_cfg.get("jinja")
    jinja_file = chat_cfg.get("jinja_file")

    if source == "inline" and jinja:
        tokenizer.chat_template = str(jinja)
        logger.info("已应用 inline chat template: %s", chat_cfg.get("name", "custom"))
    elif source == "jinja_file" and jinja_file:
        path = Path(str(jinja_file))
        tokenizer.chat_template = path.read_text(encoding="utf-8")
        logger.info("已从文件加载 chat template: %s", path)
    else:
        # tokenizer 优先
        if not getattr(tokenizer, "chat_template", None):
            if jinja:
                tokenizer.chat_template = str(jinja)
            elif jinja_file:
                tokenizer.chat_template = Path(str(jinja_file)).read_text(encoding="utf-8")
            else:
                tokenizer.chat_template = DEFAULT_CHAT_JINJA
            logger.info(
                "tokenizer 无 chat_template，已回退: %s",
                chat_cfg.get("name", "default"),
            )
        else:
            logger.info("使用模型自带 chat template (%s)", chat_cfg.get("name", "tokenizer"))

    return tokenizer


def render_chat(
    tokenizer: Any,
    messages: list[dict[str, str]],
    add_generation_prompt: bool = True,
) -> str:
    """训练/推理统一入口：渲染 messages → 字符串 prompt。"""
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
