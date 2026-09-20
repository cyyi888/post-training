"""Tests for report_to / run_name resolution (no GPU)."""

from omegaconf import OmegaConf

from src.utils.wandb_utils import default_run_name, resolve_report_to


def test_report_to_disabled():
    cfg = OmegaConf.create(
        {
            "report_to": "none",
            "wandb": {"enabled": False, "mode": "disabled"},
            "model": {"name": "Qwen/Qwen2.5-7B-Instruct"},
            "training": {"algorithm": "dpo"},
        }
    )
    assert resolve_report_to(cfg) == "none"


def test_report_to_wandb_online():
    cfg = OmegaConf.create(
        {
            "report_to": "none",
            "wandb": {"enabled": True, "mode": "online"},
            "model": {"name": "Qwen/Qwen2.5-7B-Instruct"},
            "training": {"algorithm": "dpo"},
        }
    )
    assert resolve_report_to(cfg) == "wandb"


def test_default_run_name():
    cfg = OmegaConf.create(
        {
            "run_name": None,
            "wandb": {"run_name": None},
            "model": {"name": "Qwen/Qwen2.5-7B-Instruct"},
            "training": {"algorithm": "sft"},
        }
    )
    assert default_run_name(cfg) == "sft-Qwen2.5-7B-Instruct"
