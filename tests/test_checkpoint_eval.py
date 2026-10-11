"""【设备：CPU】统一 Checkpoint 评测入口单测。

不下载大模型、不跑真实推理。通过 Mock / 依赖注入验证：
checkpoint 选择、Full/LoRA 加载分支、任务选择、Evaluator 调用、报告生成与非法路径报错。

真实权重加载与生成正确性：GPU_PENDING。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.evaluators.checkpoint import (
    CheckpointError,
    detect_checkpoint_kind,
    load_checkpoint_for_eval,
    resolve_checkpoint,
)
from src.evaluators.result import EvaluationResult
from src.evaluators.service import evaluate, resolve_eval_task


def _cfg(**kwargs):
    base = {
        "output_dir": "outputs",
        "use_gpu": False,
        "seed": 0,
        "model": SimpleNamespace(
            name="Qwen/Qwen2.5-0.5B-Instruct",
            torch_dtype=None,
            trust_remote_code=True,
            use_lora=True,
        ),
        "data": SimpleNamespace(name="gsm8k"),
        "eval": {"task": "gsm8k", "checkpoint": None, "save_report": True, "report_name": "eval_report.json"},
        "chat_template": None,
        "get": None,
    }

    class Cfg(dict):
        def __getattr__(self, item):
            try:
                return self[item]
            except KeyError as e:
                raise AttributeError(item) from e

        def get(self, key, default=None):
            return dict.get(self, key, default)

    cfg = Cfg(base)
    cfg.update(kwargs)
    return cfg


def _make_lora_ckpt(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "adapter_config.json").write_text('{"peft_type": "LORA"}', encoding="utf-8")
    (root / "adapter_model.safetensors").write_bytes(b"fake")
    return root


def _make_full_ckpt(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text('{"model_type": "qwen2"}', encoding="utf-8")
    return root


# ----- resolve / detect -----


def test_detect_lora_and_full(tmp_path: Path):
    lora = _make_lora_ckpt(tmp_path / "lora")
    full = _make_full_ckpt(tmp_path / "full")
    assert detect_checkpoint_kind(lora) == "lora"
    assert detect_checkpoint_kind(full) == "full"


def test_detect_invalid_raises(tmp_path: Path):
    bad = tmp_path / "empty"
    bad.mkdir()
    with pytest.raises(CheckpointError, match="无法识别"):
        detect_checkpoint_kind(bad)
    with pytest.raises(CheckpointError, match="不存在"):
        detect_checkpoint_kind(tmp_path / "missing")


def test_resolve_explicit_path(tmp_path: Path):
    ckpt = _make_lora_ckpt(tmp_path / "checkpoints" / "sft" / "checkpoint-3")
    cfg = _cfg(output_dir=str(tmp_path))
    assert resolve_checkpoint(cfg, str(ckpt)) == ckpt


def test_resolve_stage_alias_latest(tmp_path: Path):
    _make_lora_ckpt(tmp_path / "checkpoints" / "sft" / "checkpoint-1")
    latest = _make_lora_ckpt(tmp_path / "checkpoints" / "sft" / "checkpoint-10")
    cfg = _cfg(output_dir=str(tmp_path))
    assert resolve_checkpoint(cfg, "sft") == latest


def test_resolve_from_eval_config(tmp_path: Path):
    ckpt = _make_full_ckpt(tmp_path / "my_full")
    cfg = _cfg(output_dir=str(tmp_path), eval={"task": "gsm8k", "checkpoint": str(ckpt)})
    assert resolve_checkpoint(cfg, None) == ckpt


def test_resolve_missing_raises():
    cfg = _cfg(eval={"task": "gsm8k", "checkpoint": None})
    with pytest.raises(CheckpointError, match="未指定 checkpoint"):
        resolve_checkpoint(cfg, None)


def test_resolve_missing_stage_raises(tmp_path: Path):
    cfg = _cfg(output_dir=str(tmp_path))
    with pytest.raises(CheckpointError, match="无可用 checkpoint"):
        resolve_checkpoint(cfg, "dpo")


# ----- load branches (injected) -----


def test_load_full_calls_loader_with_ckpt_path(tmp_path: Path):
    ckpt = _make_full_ckpt(tmp_path / "full")
    calls: list[str] = []

    def fake_load(name, **kwargs):
        calls.append(name)
        return ("FULL_MODEL", "TOK")

    cfg = _cfg()
    model, tok, kind = load_checkpoint_for_eval(cfg, ckpt, load_model_fn=fake_load)
    assert kind == "full"
    assert model == "FULL_MODEL"
    assert tok == "TOK"
    assert calls == [str(ckpt)]


def test_load_lora_loads_base_then_adapter(tmp_path: Path):
    ckpt = _make_lora_ckpt(tmp_path / "adapter")
    calls: list[tuple] = []

    def fake_load(name, **kwargs):
        calls.append(("load", name))
        return (SimpleNamespace(name="base"), "BASE_TOK")

    def fake_adapter(model, path):
        calls.append(("adapter", path, model.name))
        return SimpleNamespace(name="lora_wrapped", base=model)

    cfg = _cfg()
    model, tok, kind = load_checkpoint_for_eval(
        cfg, ckpt, load_model_fn=fake_load, load_adapter_fn=fake_adapter
    )
    assert kind == "lora"
    assert model.name == "lora_wrapped"
    assert tok == "BASE_TOK"
    assert calls[0] == ("load", "Qwen/Qwen2.5-0.5B-Instruct")
    assert calls[1][0] == "adapter"
    assert calls[1][1] == str(ckpt)


# ----- evaluate orchestration -----


class _FakeEvaluator:
    def __init__(self, cfg, model, tokenizer):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer

    def evaluate(self, dataset):
        assert dataset == [{"id": 1}]
        return {"accuracy": 0.5, "format_rate": 1.0}


def test_evaluate_end_to_end_mocked(tmp_path: Path):
    ckpt = _make_lora_ckpt(tmp_path / "ckpts" / "checkpoint-1")
    report_dir = tmp_path / "reports"
    cfg = _cfg(
        output_dir=str(tmp_path),
        eval={
            "task": "gsm8k",
            "checkpoint": str(ckpt),
            "save_report": True,
            "report_name": "eval_report.json",
        },
    )

    def fake_load(name, **kwargs):
        return ("M", "T")

    def fake_adapter(model, path):
        return ("M_LORA", path)

    built = {}

    def fake_build_eval(task, config, model, tokenizer):
        built["task"] = task
        built["model"] = model
        built["tokenizer"] = tokenizer
        return _FakeEvaluator(config, model, tokenizer)

    result = evaluate(
        cfg,
        checkpoint=str(ckpt),
        dataset=[{"id": 1}],
        load_model_fn=fake_load,
        load_adapter_fn=fake_adapter,
        build_evaluator_fn=fake_build_eval,
        report_dir=report_dir,
    )

    assert isinstance(result, EvaluationResult)
    assert result.task == "gsm8k"
    assert result.checkpoint_kind == "lora"
    assert result.metrics["accuracy"] == 0.5
    assert result.validation_status == "GPU_PENDING"
    assert built["task"] == "gsm8k"
    assert built["model"] == ("M_LORA", str(ckpt))

    report_path = Path(result.report_path)
    assert report_path.is_file()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["task"] == "gsm8k"
    assert payload["checkpoint_kind"] == "lora"
    assert payload["metrics"]["accuracy"] == 0.5
    assert payload["validation_status"] == "GPU_PENDING"


def test_evaluate_selects_humaneval_task(tmp_path: Path):
    ckpt = _make_full_ckpt(tmp_path / "full")
    cfg = _cfg(
        eval={"task": "human_eval", "checkpoint": str(ckpt), "save_report": False},
        data=SimpleNamespace(name="humaneval"),
    )

    seen = {}

    class Ev:
        def __init__(self, *a, **k):
            pass

        def evaluate(self, ds):
            return {"pass_at_1": 0.0}

    def fake_build(task, config, model, tokenizer):
        seen["task"] = task
        return Ev()

    result = evaluate(
        cfg,
        dataset=[],
        load_model_fn=lambda *a, **k: ("m", "t"),
        build_evaluator_fn=fake_build,
        save_report_flag=False,
    )
    assert seen["task"] == "humaneval"
    assert result.task == "humaneval"


def test_resolve_eval_task_aliases():
    assert resolve_eval_task(_cfg(eval={"task": "gsm-8k"})) == "gsm8k"
    assert resolve_eval_task(_cfg(eval={"task": "openai_humaneval"})) == "humaneval"


def test_evaluate_invalid_checkpoint_raises(tmp_path: Path):
    cfg = _cfg(output_dir=str(tmp_path))
    with pytest.raises(CheckpointError):
        evaluate(cfg, checkpoint=str(tmp_path / "nope"))
