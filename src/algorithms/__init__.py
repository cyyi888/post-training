"""Custom post-training algorithms: DPO/IPO loss, native GRPO, reward engine.

子模块按需导入，避免 ``import src.algorithms.reward_functions`` 时连带加载 torch。
"""

__all__ = [
    "DPOLossConfig",
    "compute_preference_loss",
    "dpo_loss",
    "ipo_loss",
    "DynamicBetaController",
    "compute_grpo_loss",
    "RewardEngine",
    "build_reward_funcs",
    "GRPORolloutPipeline",
    "RolloutConfig",
    "build_sft_sampler",
    "LengthGroupedSampler",
    "build_grad_clipper",
]

_DPO_LOSS_EXPORTS = {
    "DPOLossConfig",
    "compute_preference_loss",
    "dpo_loss",
    "ipo_loss",
}


def __getattr__(name: str):
    if name in _DPO_LOSS_EXPORTS:
        from . import dpo_losses as _dpo

        return getattr(_dpo, name)
    if name == "DynamicBetaController":
        from .dynamic_beta import DynamicBetaController

        return DynamicBetaController
    if name == "compute_grpo_loss":
        from .grpo_loss import compute_grpo_loss

        return compute_grpo_loss
    if name in ("RewardEngine", "build_reward_funcs"):
        from .reward_functions import RewardEngine, build_reward_funcs

        return {
            "RewardEngine": RewardEngine,
            "build_reward_funcs": build_reward_funcs,
        }[name]
    if name in ("GRPORolloutPipeline", "RolloutConfig"):
        from .rollout import GRPORolloutPipeline, RolloutConfig

        return {
            "GRPORolloutPipeline": GRPORolloutPipeline,
            "RolloutConfig": RolloutConfig,
        }[name]
    if name in ("build_sft_sampler", "LengthGroupedSampler"):
        from .sft_samplers import LengthGroupedSampler, build_sft_sampler

        return {
            "build_sft_sampler": build_sft_sampler,
            "LengthGroupedSampler": LengthGroupedSampler,
        }[name]
    if name == "build_grad_clipper":
        from .grad_clip import build_grad_clipper

        return build_grad_clipper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
