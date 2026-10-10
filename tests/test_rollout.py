"""【设备：CPU】GRPO Rollout 管线逻辑单测（假 tokenizer / mock generate）。

原因：覆盖分桶、动态 batch、格式校验、拒绝采样与组内组装，用假模型即可验证
控制流与指标字段，不加载 Qwen、无需 CUDA。真实 generate + 引擎短训见
``tests/test_grpo_pipeline.py``（GPU）。
"""

from __future__ import annotations

from typing import Any

import torch

from src.algorithms.reward_functions import RewardEngine
from src.algorithms.rollout import (
    GRPORolloutPipeline,
    RolloutConfig,
    assign_bucket_id,
    iter_dynamic_batches,
    order_rows_by_length,
    resolve_bucket_batch_sizes,
    validate_completion,
)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    pad_token = "<pad>"
    eos_token = "<eos>"
    padding_side = "left"
    chat_template = None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [10 + (ord(c) % 50) for c in text] or [10]

    def __call__(self, texts, return_tensors="pt", padding=True):
        if isinstance(texts, str):
            texts = [texts]
        encoded = [self.encode(t) for t in texts]
        max_len = max(len(e) for e in encoded)
        input_ids = []
        attn = []
        for e in encoded:
            pad = max_len - len(e)
            input_ids.append([self.pad_token_id] * pad + e)
            attn.append([0] * pad + [1] * len(e))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }

    def decode(self, ids, skip_special_tokens=True):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if 42 in ids:
            return r"reasoning \boxed{42}"
        if 44 in ids:
            return "step\n#### 42"
        if 43 in ids:
            return "garbage without box"
        return ""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = [str(m.get("content", "")) for m in messages]
        text = "\n".join(parts)
        if add_generation_prompt:
            text += "\nAssistant:"
        if tokenize:
            return self.encode(text)
        return text


class _FakePolicyModel:
    """假策略：避免 MagicMock 自动 ``.module`` 干扰 DDP 解包。"""

    def __init__(self, completion_token: int = 42):
        self._param = torch.nn.Parameter(torch.zeros(1))
        self.training = True
        self.completion_token = completion_token
        self.last_generate_kwargs: dict[str, Any] = {}

    def parameters(self):
        yield self._param

    def train(self, mode: bool = True):
        self.training = bool(mode)
        return self

    def eval(self):
        self.training = False
        return self

    def generate(self, **kwargs):
        self.last_generate_kwargs = dict(kwargs)
        input_ids = kwargs["input_ids"]
        b = input_ids.shape[0]
        nret = int(kwargs.get("num_return_sequences", 1))
        max_new = int(kwargs.get("max_new_tokens", 8))
        rows = []
        for i in range(b):
            for _ in range(nret):
                prompt = input_ids[i]
                comp = torch.full((max_new,), self.completion_token, dtype=torch.long)
                comp[-1] = 1
                rows.append(torch.cat([prompt, comp], dim=0))
        return torch.stack(rows, dim=0)


def _fake_model(completion_token: int = 42):
    return _FakePolicyModel(completion_token=completion_token)


def test_validate_completion_answer_formats_and_length():
    ok, reason = validate_completion(
        r"\boxed{1}",
        torch.tensor([1] * 20),
        require_answer_format=True,
        filter_extreme_length=True,
        min_tokens=4,
        max_tokens=64,
    )
    assert ok and reason == ""
    ok2, _ = validate_completion(
        "ans\n#### 3",
        torch.tensor([1] * 20),
        require_answer_format=True,
        filter_extreme_length=False,
    )
    assert ok2
    bad, reason = validate_completion(
        "nope",
        torch.tensor([1] * 20),
        require_answer_format=True,
        filter_extreme_length=False,
    )
    assert not bad and reason == "missing_answer_format"
    short, reason = validate_completion(
        r"\boxed{1}",
        torch.tensor([1, 2]),
        require_answer_format=True,
        filter_extreme_length=True,
        min_tokens=10,
        max_tokens=64,
    )
    assert not short and reason == "too_short"


def test_assign_bucket_and_dynamic_batch_sizes():
    assert assign_bucket_id(50, [128, 256, 512]) == 0
    assert assign_bucket_id(200, [128, 256, 512]) == 1
    assert assign_bucket_id(300, [128, 256, 512]) == 2
    assert assign_bucket_id(800, [128, 256, 512]) == 3
    sizes = resolve_bucket_batch_sizes(2, 4)
    assert sizes[0] >= sizes[1] >= sizes[-1]
    assert sizes[-1] == 1


def test_iter_dynamic_batches_short_larger_than_long():
    rows = [{"id": i, "prompt": [{"role": "user", "content": "x"}]} for i in range(8)]
    lengths = [10, 20, 30, 40, 200, 220, 240, 260]
    batches = list(
        iter_dynamic_batches(
            rows,
            lengths,
            base_batch_size=1,
            boundaries=[128, 256, 512],
            bucket_batch_sizes=[4, 2, 1, 1],
            bucket=True,
            seed=0,
        )
    )
    assert batches
    short_batches = [b for b in batches if all(r["id"] < 4 for r in b)]
    long_batches = [b for b in batches if all(r["id"] >= 4 for r in b)]
    assert short_batches and max(len(b) for b in short_batches) == 4
    assert long_batches and max(len(b) for b in long_batches) <= 2


def test_order_rows_by_length_uses_buckets():
    rows = [
        {"prompt": [{"role": "user", "content": "a"}], "ground_truth": "1"},
        {"prompt": [{"role": "user", "content": "b" * 200}], "ground_truth": "2"},
    ]
    lengths = [5, 200]
    ordered = order_rows_by_length(
        rows, lengths, batch_size=1, bucket=True, seed=0, boundaries=[128, 256]
    )
    assert len(ordered) == 2


def test_pipeline_assembles_advantages_and_metrics():
    tok = FakeTokenizer()
    model = _fake_model(completion_token=42)
    reward = RewardEngine(
        ["accuracy", "format", "length"],
        weights=[1.0, 0.2, 0.1],
        length_reward={"min_tokens": 1, "max_tokens": 64, "target_min": 1, "target_max": 64},
    )
    logged: list[dict[str, Any]] = []

    def log_fn(m, step=None):
        logged.append(dict(m))

    cfg = RolloutConfig(
        num_generations=2,
        max_completion_length=8,
        max_resample=0,
        max_group_resample=0,
        min_valid_per_group=2,
        require_answer_format=True,
        filter_extreme_length=False,
        top_p=0.9,
        top_k=50,
        clear_cache=False,
    )
    pipe = GRPORolloutPipeline(reward, cfg, log_fn=log_fn)
    rows = [{"prompt": [{"role": "user", "content": "q1"}], "ground_truth": "42"}]
    batch = pipe.run(rows, model, tok, step=1)
    assert batch is not None and not batch.empty
    assert batch.rewards.numel() == 2
    assert abs(float(batch.advantages.mean())) < 1e-5
    assert batch.prompt_ids and batch.answer_ids
    assert "reward/max" in batch.metrics
    assert "perf/tokens_per_sec" in batch.metrics
    assert model.last_generate_kwargs.get("top_p") == 0.9
    assert model.last_generate_kwargs.get("top_k") == 50
    assert logged and "rollout/reward/mean" in logged[0]


def test_pipeline_accepts_hash_format():
    tok = FakeTokenizer()
    model = _fake_model(completion_token=44)
    reward = RewardEngine(["format"], weights=[1.0])
    cfg = RolloutConfig(
        num_generations=2,
        max_completion_length=8,
        max_resample=0,
        max_group_resample=0,
        min_valid_per_group=2,
        require_answer_format=True,
        filter_extreme_length=False,
        clear_cache=False,
    )
    pipe = GRPORolloutPipeline(reward, cfg)
    batch = pipe.run(
        [{"prompt": [{"role": "user", "content": "q"}], "ground_truth": "42"}],
        model,
        tok,
    )
    assert batch is not None and not batch.empty


def test_pipeline_rejects_and_drops_group():
    tok = FakeTokenizer()
    model = _fake_model(completion_token=43)
    reward = RewardEngine(["format"], weights=[1.0])
    cfg = RolloutConfig(
        num_generations=2,
        max_completion_length=4,
        max_resample=1,
        max_group_resample=1,
        min_valid_per_group=2,
        require_answer_format=True,
        filter_extreme_length=False,
        clear_cache=False,
    )
    pipe = GRPORolloutPipeline(reward, cfg)
    batch = pipe.run(
        [{"prompt": [{"role": "user", "content": "q"}], "ground_truth": "0"}],
        model,
        tok,
    )
    assert batch is not None and batch.empty
    assert batch.metrics["reject_rate"] == 1.0
    assert batch.metrics["group_resample_attempts"] >= 1.0
    assert batch.metrics["resample_attempts"] >= 2.0


def test_pipeline_group_resample_when_below_half():
    tok = FakeTokenizer()

    class HalfThenGood:
        def __init__(self):
            self.param = torch.nn.Parameter(torch.zeros(1))
            self.training = True
            self._calls = 0

        def parameters(self):
            return iter([self.param])

        def train(self, mode=True):
            self.training = bool(mode)

        def eval(self):
            self.training = False

        def generate(self, **kwargs):
            self._calls += 1
            input_ids = kwargs["input_ids"]
            b = input_ids.shape[0]
            nret = int(kwargs.get("num_return_sequences", 1))
            max_new = int(kwargs.get("max_new_tokens", 4))
            rows = []
            for i in range(b):
                for _ in range(nret):
                    tok_id = 43 if self._calls == 1 else 42
                    comp = torch.full((max_new,), tok_id, dtype=torch.long)
                    comp[-1] = 1
                    rows.append(torch.cat([input_ids[i], comp], dim=0))
            return torch.stack(rows, dim=0)

    model = HalfThenGood()
    reward = RewardEngine(["format"], weights=[1.0])
    cfg = RolloutConfig(
        num_generations=4,
        max_completion_length=4,
        max_resample=0,
        max_group_resample=2,
        min_valid_per_group=2,
        require_answer_format=True,
        filter_extreme_length=False,
        clear_cache=False,
    )
    pipe = GRPORolloutPipeline(reward, cfg)
    batch = pipe.run(
        [{"prompt": [{"role": "user", "content": "q"}], "ground_truth": "x"}],
        model,
        tok,
    )
    assert batch is not None and not batch.empty
    assert batch.metrics["group_resample_attempts"] >= 1.0
    assert batch.group_sizes == [4]
