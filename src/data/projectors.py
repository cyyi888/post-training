"""Sample → 各训练算法所需字段投影。"""

from __future__ import annotations

from typing import Any

from datasets import Dataset, DatasetDict

from src.data.schema import Sample


def project_sft(samples: list[Sample]) -> Dataset:
    rows = [{"messages": s.to_messages(include_assistant=True)} for s in samples]
    return Dataset.from_list(rows) if rows else Dataset.from_list([])


def project_dpo(samples: list[Sample]) -> Dataset:
    rows = []
    for s in samples:
        if not s.chosen or not s.rejected:
            continue
        prompt_msgs = []
        if s.system:
            prompt_msgs.append({"role": "system", "content": s.system})
        prompt_msgs.append({"role": "user", "content": s.prompt})
        rows.append(
            {
                "prompt": prompt_msgs,
                "chosen": [{"role": "assistant", "content": s.chosen}],
                "rejected": [{"role": "assistant", "content": s.rejected}],
            }
        )
    return Dataset.from_list(rows) if rows else Dataset.from_list([])


def project_grpo(samples: list[Sample]) -> Dataset:
    """投影 GRPO / PPO 在线 RL 字段。

    除 ``prompt`` / ``ground_truth`` 外，透传 HumanEval 的 ``test`` / ``entry_point``
    与原始 ``task_prompt``（代码题函数签名），供 ``code`` 奖励执行测例。
    """
    rows = []
    for s in samples:
        prompt = []
        if s.system:
            prompt.append({"role": "system", "content": s.system})
        prompt.append({"role": "user", "content": s.prompt})
        row: dict[str, Any] = {
            "prompt": prompt,
            "ground_truth": s.metadata.get("ground_truth"),
            "task_prompt": s.prompt,
            "source_format": s.metadata.get("source_format"),
        }
        # HumanEval / 代码题元数据（缺失时为 None，数学题不受影响）
        if "entry_point" in s.metadata:
            row["entry_point"] = s.metadata.get("entry_point")
        if "test" in s.metadata:
            row["test"] = s.metadata.get("test")
        rows.append(row)
    return Dataset.from_list(rows) if rows else Dataset.from_list([])


_PROJECTORS = {
    "sft": project_sft,
    "dpo": project_dpo,
    "grpo": project_grpo,
}


def project_splits(
    splits: dict[str, list[Sample]],
    algorithm: str,
) -> DatasetDict:
    if algorithm not in _PROJECTORS:
        raise KeyError(f"无投影器 '{algorithm}'，可用: {list(_PROJECTORS)}")
    fn = _PROJECTORS[algorithm]
    mapping = {}
    # HF Trainer 习惯用 train / eval
    key_map = {"train": "train", "val": "eval", "test": "test"}
    for src, dst in key_map.items():
        if src in splits:
            mapping[dst] = fn(splits[src])
    if "eval" not in mapping and "test" in mapping:
        mapping["eval"] = mapping["test"]
    return DatasetDict(mapping)
