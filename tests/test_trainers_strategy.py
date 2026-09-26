"""策略模式：BaseTrainer 接口与工厂基础测试（不依赖 GPU）。"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from src.core.base import BaseTrainer, TrainResult
from src.core.registry import ALGORITHM_REGISTRY


def test_resolve_splits_from_dict():
    train, ev = BaseTrainer.resolve_splits({"train": [1, 2], "eval": [3]})
    assert train == [1, 2]
    assert ev == [3]


def test_resolve_splits_single_dataset():
    train, ev = BaseTrainer.resolve_splits(["a", "b"])
    assert train == ["a", "b"]
    assert ev is None


def test_resolve_splits_rejects_none():
    with pytest.raises(ValueError):
        BaseTrainer.resolve_splits(None)


def _ensure_strategies_registered():
    # 轻量注册：仅 PPO 不依赖 trl；其余在缺 trl 时 skip
    import src.trainers.ppo_trainer  # noqa: F401

    try:
        import src.trainers.dpo_trainer  # noqa: F401
        import src.trainers.grpo_trainer  # noqa: F401
        import src.trainers.sft_trainer  # noqa: F401
    except ImportError:
        pytest.skip("trl / transformers 未安装，跳过完整策略注册测试")


def test_all_strategies_expose_train_config_dataset():
    _ensure_strategies_registered()
    from src.trainers.factory import available_algorithms, get_trainer

    algos = available_algorithms()
    assert "ppo" in algos
    for name in algos:
        trainer = get_trainer(name)
        assert isinstance(trainer, BaseTrainer)
        assert trainer.name == name
        sig = inspect.signature(trainer.train)
        params = list(sig.parameters.keys())
        assert params[:2] == ["config", "dataset"], f"{name}.train 签名应为 train(config, dataset)"


def test_ppo_is_trl_baseline():
    import src.trainers.ppo_trainer as ppo_mod
    from src.trainers.factory import get_trainer

    trainer = get_trainer("ppo")
    assert trainer.name == "ppo"
    assert "TRL" in ppo_mod.PPOStageTrainer.__doc__


def test_get_trainer_unknown():
    from src.trainers.factory import get_trainer

    with pytest.raises(KeyError, match="未知训练策略"):
        get_trainer("no_such_algo")


def test_train_result_shape():
    r = TrainResult(metrics={"loss": 1.0}, checkpoint_dir="out")
    assert r.metrics["loss"] == 1.0
    assert r.checkpoint_dir == "out"


class _Cfg:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def get(self, key, default=None):
        return self.__dict__.get(key, default)


def test_latest_checkpoint_picks_highest_step(tmpdir):
    root = Path(str(tmpdir))
    (root / "checkpoint-2").mkdir()
    (root / "checkpoint-10").mkdir()
    assert BaseTrainer.latest_checkpoint(str(root)).endswith("checkpoint-10")


def test_distributed_none_ignores_env(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "2")

    class Toy(BaseTrainer):
        name = "toy"

        def run_algorithm(self, ctx):
            raise AssertionError("不应调用算法")

    trainer = Toy()
    trainer.detect_distributed(_Cfg(engineering=_Cfg(distributed="none")))
    assert trainer.world_size == 1
    assert trainer.is_main_process is True


def test_template_train_calls_algorithm_only(tmpdir):
    class Toy(BaseTrainer):
        name = "toy"

        def build_context(self, config, dataset):
            from src.core.base import TrainContext

            return TrainContext(
                config=config,
                dataset=dataset,
                train_dataset=dataset,
                eval_dataset=None,
                model=None,
                tokenizer=None,
                checkpoint_dir="toy-ckpt",
            )

        def run_algorithm(self, ctx):
            assert ctx.dataset == {"train": [1]}
            return TrainResult(metrics={"loss": 0.2}, checkpoint_dir=ctx.checkpoint_dir)

    cfg = _Cfg(
        seed=0,
        output_dir=str(tmpdir),
        logging=_Cfg(
            level="INFO",
            to_console=False,
            to_file=True,
            filename="run.log",
            format="%(message)s",
            datefmt="%H:%M:%S",
            quiet_libs=[],
        ),
        wandb=_Cfg(enabled=False, mode="disabled"),
        engineering=_Cfg(distributed="none", resume_from_checkpoint=None),
        training=_Cfg(algorithm="toy"),
        model=_Cfg(name="toy"),
    )
    result = Toy().train(cfg, {"train": [1]})
    assert result.metrics["loss"] == 0.2
    assert result.checkpoint_dir == "toy-ckpt"


def _toy():
    class Toy(BaseTrainer):
        name = "toy"

        def run_algorithm(self, ctx):
            return TrainResult()

    return Toy()


def test_runtime_falls_back_to_training_yaml():
    cfg = _Cfg(
        seed=7,
        engineering=_Cfg(
            gradient_accumulation_steps=None,
            mixed_precision="no",
            gradient_checkpointing=None,
            lr_scheduler_type=None,
            warmup_ratio=None,
            learning_rate=None,
        ),
        training=_Cfg(
            learning_rate=8e-5,
            gradient_accumulation_steps=8,
            gradient_checkpointing=False,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
        ),
    )
    runtime = _toy().resolve_runtime(cfg)
    assert runtime.seed == 7
    assert runtime.learning_rate == 8e-5
    assert runtime.gradient_accumulation_steps == 8
    assert runtime.gradient_checkpointing is False
    assert runtime.warmup_ratio == 0.03
    assert runtime.lr_scheduler_type == "cosine"
    assert runtime.mixed_precision == "no"


def test_runtime_engineering_overrides_training():
    cfg = _Cfg(
        seed=1,
        engineering=_Cfg(
            gradient_accumulation_steps=2,
            learning_rate=1e-4,
            mixed_precision="fp16",
            gradient_checkpointing=True,
            lr_scheduler_type="linear",
            warmup_ratio=0.1,
        ),
        training=_Cfg(
            learning_rate=8e-5,
            gradient_accumulation_steps=8,
            gradient_checkpointing=False,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
        ),
    )
    runtime = _toy().resolve_runtime(cfg)
    assert runtime.learning_rate == 1e-4
    assert runtime.gradient_accumulation_steps == 2
    assert runtime.gradient_checkpointing is True
    assert runtime.lr_scheduler_type == "linear"
    assert runtime.warmup_ratio == 0.1
    assert runtime.mixed_precision == "fp16"


def test_estimate_update_steps():
    assert BaseTrainer.estimate_update_steps(10, 2, 1, 4) == 2
    assert BaseTrainer.estimate_update_steps(1, 1, 1, 1) == 1


def test_hf_kwargs_carry_runtime_options():
    pytest.importorskip("omegaconf")
    trainer = _toy()
    cfg = _Cfg(
        seed=3,
        output_dir="outputs",
        report_to="none",
        model=_Cfg(name="toy"),
        training=_Cfg(
            algorithm="toy",
            learning_rate=1e-5,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            warmup_ratio=0.0,
            lr_scheduler_type="constant",
        ),
        engineering=_Cfg(mixed_precision="no", distributed="none"),
    )
    trainer.runtime = trainer.resolve_runtime(cfg)
    kwargs = trainer.hf_training_kwargs(cfg)
    assert kwargs["gradient_accumulation_steps"] == 4
    assert kwargs["gradient_checkpointing"] is True
    assert kwargs["lr_scheduler_type"] == "constant"
    assert kwargs["seed"] == 3
    assert kwargs["fp16"] is False
    assert kwargs["bf16"] is False


def test_scheduler_warmup_then_decay():
    torch = pytest.importorskip("torch")
    from src.core.base import RuntimeOptions

    trainer = _toy()
    trainer.runtime = RuntimeOptions(
        learning_rate=1.0,
        warmup_ratio=0.5,
        lr_scheduler_type="linear",
        mixed_precision="no",
    )
    model = torch.nn.Linear(2, 2)
    _opt, scheduler, scaler = trainer.build_optimizer_and_scheduler(model, num_update_steps=4)
    assert scaler is None
    assert abs(scheduler.get_last_lr()[0] - 0.5) < 1e-5
    scheduler.step()
    assert abs(scheduler.get_last_lr()[0] - 1.0) < 1e-5
    scheduler.step()
    assert scheduler.get_last_lr()[0] < 1.0
