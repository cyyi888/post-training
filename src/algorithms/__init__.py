"""Custom post-training algorithms: DPO/IPO loss, native GRPO, reward engine.

子模块按需导入，避免 ``import src.algorithms.reward_functions`` 时连带加载 torch。
"""

__all__ = [
    "DPOLoss",
    "compute_dpo_loss",
    "compute_variant_loss",
    "dpo_loss",
    "ipo_loss",
    "DynamicBetaController",
    "compute_grpo_loss",
    "RewardEngine",
    "build_reward_funcs",
    "build_sft_sampler",
    "LengthGroupedSampler",
    "build_grad_clipper",
]


def __getattr__(name: str):
    if name in ("DPOLoss", "compute_dpo_loss"):
        from .dpo_loss import DPOLoss, compute_dpo_loss

        return {"DPOLoss": DPOLoss, "compute_dpo_loss": compute_dpo_loss}[name]
    if name in ("dpo_loss", "ipo_loss"):
        from .dpo_losses import dpo_loss, ipo_loss

        return {"dpo_loss": dpo_loss, "ipo_loss": ipo_loss}[name]
    if name == "DynamicBetaController":
        from .dynamic_beta import DynamicBetaController

        return DynamicBetaController
    if name == "compute_variant_loss":
        from .dpo_variants import compute_variant_loss

        return compute_variant_loss
    if name == "compute_grpo_loss":
        from .grpo_loss import compute_grpo_loss

        return compute_grpo_loss
    if name in ("RewardEngine", "build_reward_funcs"):
        from .reward_functions import RewardEngine, build_reward_funcs

        return {
            "RewardEngine": RewardEngine,
            "build_reward_funcs": build_reward_funcs,
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
