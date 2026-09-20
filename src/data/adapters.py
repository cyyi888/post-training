"""外部格式 → Sample 适配器。"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any

from src.data.schema import Sample

ADAPTER_REGISTRY: dict[str, type] = {}


def register_adapter(name: str):
    def deco(cls: type[BaseAdapter]):
        ADAPTER_REGISTRY[name] = cls
        return cls

    return deco


class BaseAdapter(ABC):
    @abstractmethod
    def convert(self, raw: dict[str, Any], **kwargs: Any) -> Sample:
        ...

    def convert_many(self, rows: list[dict[str, Any]], **kwargs: Any) -> list[Sample]:
        return [self.convert(r, **kwargs) for r in rows]


def get_adapter(name: str) -> BaseAdapter:
    if name not in ADAPTER_REGISTRY:
        raise KeyError(f"未知格式适配器 '{name}'，可用: {list(ADAPTER_REGISTRY)}")
    return ADAPTER_REGISTRY[name]()


@register_adapter("instruction")
class InstructionAdapter(BaseAdapter):
    """Alpaca 风格：instruction (+ input) → prompt，output → chosen。"""

    def convert(self, raw: dict[str, Any], **kwargs: Any) -> Sample:
        instruction = str(raw.get("instruction") or raw.get("query") or "")
        inp = str(raw.get("input") or "").strip()
        prompt = f"{instruction}\n{inp}".strip() if inp else instruction
        system = kwargs.get("system_prompt") or raw.get("system")
        return Sample(
            prompt=prompt,
            chosen=str(raw.get("output") or raw.get("response") or "") or None,
            system=system,
            metadata={"source_format": "instruction", **(raw.get("metadata") or {})},
        )


@register_adapter("sharegpt")
class ShareGPTAdapter(BaseAdapter):
    """ShareGPT / conversations：取首轮 user 为 prompt，首轮 assistant 为 chosen。"""

    def convert(self, raw: dict[str, Any], **kwargs: Any) -> Sample:
        conv = raw.get("conversations") or raw.get("messages") or []
        system = kwargs.get("system_prompt") or raw.get("system")
        user_parts: list[str] = []
        assistant_parts: list[str] = []
        for turn in conv:
            role = (turn.get("from") or turn.get("role") or "").lower()
            content = str(turn.get("value") or turn.get("content") or "")
            if role in ("system",):
                system = system or content
            elif role in ("human", "user"):
                user_parts.append(content)
            elif role in ("gpt", "assistant"):
                assistant_parts.append(content)
        prompt = user_parts[0] if user_parts else ""
        chosen = assistant_parts[0] if assistant_parts else None
        return Sample(
            prompt=prompt,
            chosen=chosen,
            system=system,
            metadata={
                "source_format": "sharegpt",
                "n_user_turns": len(user_parts),
                "n_assistant_turns": len(assistant_parts),
            },
        )


@register_adapter("dpo_pair")
class DPOPairAdapter(BaseAdapter):
    """偏好对：prompt + chosen + rejected（支持 messages 列表或纯文本）。"""

    def convert(self, raw: dict[str, Any], **kwargs: Any) -> Sample:
        system = kwargs.get("system_prompt") or raw.get("system")
        prompt, chosen, rejected = self._extract_triple(raw)
        if system is None and isinstance(raw.get("chosen"), list):
            system = next(
                (m.get("content") for m in raw["chosen"] if m.get("role") == "system"),
                None,
            )
        return Sample(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            system=system,
            metadata={"source_format": "dpo_pair"},
        )

    @staticmethod
    def _extract_triple(raw: dict[str, Any]) -> tuple[str, str | None, str | None]:
        # 纯文本字段
        if isinstance(raw.get("prompt"), str) and "chosen" in raw:
            chosen = raw["chosen"]
            rejected = raw.get("rejected")
            if isinstance(chosen, list):
                chosen = next(
                    (m["content"] for m in chosen if m.get("role") == "assistant"), ""
                )
            if isinstance(rejected, list):
                rejected = next(
                    (m["content"] for m in rejected if m.get("role") == "assistant"), ""
                )
            prompt = raw["prompt"]
            if isinstance(prompt, list):
                prompt = next(
                    (m["content"] for m in prompt if m.get("role") == "user"), ""
                )
            return str(prompt), (str(chosen) if chosen is not None else None), (
                str(rejected) if rejected is not None else None
            )

        # banghua / TRL 风格：chosen/rejected 为 messages
        chosen_msgs = raw.get("chosen") or []
        rejected_msgs = raw.get("rejected") or []
        prompt = ""
        chosen_text = None
        rejected_text = None
        if isinstance(chosen_msgs, list) and chosen_msgs:
            prompt = next(
                (m["content"] for m in chosen_msgs if m.get("role") == "user"), ""
            )
            chosen_text = next(
                (m["content"] for m in chosen_msgs if m.get("role") == "assistant"),
                None,
            )
        if isinstance(rejected_msgs, list) and rejected_msgs:
            rejected_text = next(
                (m["content"] for m in rejected_msgs if m.get("role") == "assistant"),
                None,
            )
            if not prompt:
                prompt = next(
                    (m["content"] for m in rejected_msgs if m.get("role") == "user"),
                    "",
                )
        return str(prompt), chosen_text, rejected_text


@register_adapter("gsm8k")
class GSM8KAdapter(BaseAdapter):
    """GSM8K：question → prompt，#### 答案写入 metadata.ground_truth。"""

    def convert(self, raw: dict[str, Any], **kwargs: Any) -> Sample:
        system = kwargs.get("system_prompt")
        answer = str(raw.get("answer", ""))
        match = re.search(r"####\s*(-?\d+)", answer)
        ground_truth = match.group(1) if match else None
        return Sample(
            prompt=str(raw.get("question", "")),
            chosen=answer or None,
            system=system,
            metadata={
                "source_format": "gsm8k",
                "ground_truth": ground_truth,
                "stratify_key": "gsm8k",
            },
        )


@register_adapter("humaneval")
class HumanEvalAdapter(BaseAdapter):
    def convert(self, raw: dict[str, Any], **kwargs: Any) -> Sample:
        return Sample(
            prompt=str(raw.get("prompt", "")),
            chosen=raw.get("canonical_solution"),
            system=kwargs.get("system_prompt"),
            metadata={
                "source_format": "humaneval",
                "ground_truth": raw.get("canonical_solution"),
                "entry_point": raw.get("entry_point"),
                "test": raw.get("test"),
                "stratify_key": "humaneval",
            },
        )
