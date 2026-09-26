"""训练策略：SFT / DPO / 在线 RL（GRPO、PPO），统一 BaseTrainer.train(config, dataset)。"""

from .factory import available_algorithms, get_trainer

__all__ = [
    "get_trainer",
    "available_algorithms",
    "SFTStageTrainer",
    "DPOStageTrainer",
    "GRPOStageTrainer",
    "PPOStageTrainer",
]


def __getattr__(name: str):
    """按需导入具体策略，避免未安装 trl 时 import 失败。"""
    if name == "SFTStageTrainer":
        from .sft_trainer import SFTStageTrainer

        return SFTStageTrainer
    if name == "DPOStageTrainer":
        from .dpo_trainer import DPOStageTrainer

        return DPOStageTrainer
    if name == "GRPOStageTrainer":
        from .grpo_trainer import GRPOStageTrainer

        return GRPOStageTrainer
    if name == "PPOStageTrainer":
        from .ppo_trainer import PPOStageTrainer

        return PPOStageTrainer
    raise AttributeError(name)
