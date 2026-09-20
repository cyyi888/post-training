"""统一中间格式 Sample：所有外部数据先转成此结构，再投影到各训练器。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Sample:
    """标准样本结构。

    - prompt: 用户侧输入（纯文本，或多轮 messages 序列化前的文本）
    - chosen: SFT 目标回答 / DPO 正例
    - rejected: DPO 负例（SFT/GRPO 可为空）
    - system: 可选系统提示
    - metadata: 来源格式、ground_truth、分层键等
    """

    prompt: str
    chosen: str | None = None
    rejected: str | None = None
    system: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str | None = None

    def __post_init__(self) -> None:
        if self.id is None:
            self.id = self.content_hash()[:16]

    def content_hash(self) -> str:
        payload = {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "system": self.system,
        }
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Sample:
        return cls(
            prompt=str(data.get("prompt", "")),
            chosen=data.get("chosen"),
            rejected=data.get("rejected"),
            system=data.get("system"),
            metadata=dict(data.get("metadata") or {}),
            id=data.get("id"),
        )

    def to_messages(self, include_assistant: bool = True) -> list[dict[str, str]]:
        """转为 chat messages，供 Chat Template / SFT 使用。"""
        messages: list[dict[str, str]] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": self.prompt})
        if include_assistant and self.chosen is not None:
            messages.append({"role": "assistant", "content": self.chosen})
        return messages
