"""数据流水线单元测试（不依赖 GPU / 外网）。"""

from __future__ import annotations

import pytest

from src.data.adapters import get_adapter
from src.data.cleaning import clean_samples, deduplicate, filter_length
from src.data.schema import Sample
from src.data.split import split_samples
from src.data.versioning import compute_samples_hash, make_version


def test_instruction_adapter():
    raw = {"instruction": "翻译", "input": "hello", "output": "你好"}
    s = get_adapter("instruction").convert(raw)
    assert "翻译" in s.prompt and "hello" in s.prompt
    assert s.chosen == "你好"
    assert s.metadata["source_format"] == "instruction"


def test_sharegpt_adapter():
    raw = {
        "conversations": [
            {"from": "human", "value": "Hi"},
            {"from": "gpt", "value": "Hello!"},
        ]
    }
    s = get_adapter("sharegpt").convert(raw)
    assert s.prompt == "Hi"
    assert s.chosen == "Hello!"


def test_dpo_pair_adapter_messages():
    raw = {
        "chosen": [
            {"role": "user", "content": "Name?"},
            {"role": "assistant", "content": "A"},
        ],
        "rejected": [
            {"role": "user", "content": "Name?"},
            {"role": "assistant", "content": "B"},
        ],
    }
    s = get_adapter("dpo_pair").convert(raw)
    assert s.prompt == "Name?"
    assert s.chosen == "A"
    assert s.rejected == "B"


def test_gsm8k_adapter():
    raw = {"question": "1+1?", "answer": "Reasoning\n#### 2"}
    s = get_adapter("gsm8k").convert(raw, system_prompt="sys")
    assert s.prompt == "1+1?"
    assert s.metadata["ground_truth"] == "2"
    assert s.system == "sys"


def test_clean_dedup_and_length():
    samples = [
        Sample(prompt="hello world", chosen="ok"),
        Sample(prompt="hello world", chosen="ok"),  # dup
        Sample(prompt="x", chosen="y"),  # too short if min_chars=8
    ]
    samples = deduplicate(samples)
    assert len(samples) == 2
    filtered = filter_length(samples, min_chars=8, max_chars=1000)
    assert len(filtered) == 1
    cleaned = clean_samples(
        [
            Sample(prompt="hello world", chosen="ok"),
            Sample(prompt="hello world", chosen="ok"),
        ],
        {"dedup": True, "min_chars": 1, "filter_low_quality": True},
    )
    assert len(cleaned) == 1


def test_split_reproducible():
    samples = [Sample(prompt=f"q{i}", chosen=f"a{i}") for i in range(20)]
    a = split_samples(samples, ratios=(0.7, 0.15, 0.15), seed=12)
    b = split_samples(samples, ratios=(0.7, 0.15, 0.15), seed=12)
    assert [s.prompt for s in a["train"]] == [s.prompt for s in b["train"]]
    assert len(a["train"]) + len(a["val"]) + len(a["test"]) == 20


def test_version_hash_stable():
    samples = [Sample(prompt="p", chosen="c"), Sample(prompt="p2", chosen="c2")]
    h1 = compute_samples_hash(samples)
    h2 = compute_samples_hash(list(reversed(samples)))
    assert h1 == h2
    ver = make_version({"train": samples, "val": [], "test": []}, version="v-test")
    assert ver.version == "v-test"
    assert ver.num_samples == 2


def test_projectors():
    datasets = pytest.importorskip("datasets")
    from src.data.projectors import project_dpo, project_grpo, project_sft

    sft_s = [Sample(prompt="q", chosen="a", system="sys")]
    dpo_s = [Sample(prompt="q", chosen="good", rejected="bad")]
    grpo_s = [Sample(prompt="q", metadata={"ground_truth": "1"})]

    sft = project_sft(sft_s)
    assert "messages" in sft[0]
    assert sft[0]["messages"][-1]["content"] == "a"

    dpo = project_dpo(dpo_s)
    assert dpo[0]["chosen"][0]["content"] == "good"

    grpo = project_grpo(grpo_s)
    assert grpo[0]["ground_truth"] == "1"
