"""训练指标字典（定义 / step 语义 / 适用算法）。

跨算法对比请优先看独立评测（GSM8K 准确率等），不要只比 train/loss。
KL 名称相近不等于定义相同，禁止盲目合并曲线。
"""

from __future__ import annotations

from typing import Any

# step_kind 取值见 metric_sink.STEP_*

METRIC_CATALOG: dict[str, dict[str, Any]] = {
    # ----- 公共（有则报，不强制）-----
    "train/loss": {
        "unit": "nat",
        "step_kind": "optimizer_step",
        "algorithms": ["sft", "dpo", "grpo", "ppo"],
        "definition": "当前优化步的标量训练损失（各算法损失定义不同，不可直接横比）。",
    },
    "train/learning_rate": {
        "unit": "1",
        "step_kind": "optimizer_step",
        "algorithms": ["sft", "dpo", "grpo", "ppo"],
        "definition": "优化器当前学习率。",
    },
    "train/epoch": {
        "unit": "1",
        "step_kind": "optimizer_step",
        "algorithms": ["sft", "dpo"],
        "definition": "HF Trainer 报告的小数 epoch。",
    },
    "eval/loss": {
        "unit": "nat",
        "step_kind": "optimizer_step",
        "algorithms": ["sft", "dpo"],
        "definition": "验证集损失（定义随算法）。",
    },
    "meta/algorithm": {
        "unit": "str",
        "step_kind": None,
        "algorithms": ["sft", "dpo", "grpo", "ppo"],
        "definition": "产生该条指标的算法名。",
    },
    "meta/step_kind": {
        "unit": "str",
        "step_kind": None,
        "algorithms": ["sft", "dpo", "grpo", "ppo"],
        "definition": "本条指标对应的 step 语义：optimizer_step | rollout_step | epoch | final。",
    },
    # ----- DPO 专属 -----
    "train/mean_kl": {
        "unit": "nat",
        "step_kind": "optimizer_step",
        "algorithms": ["dpo"],
        "definition": (
            "DPO：0.5*(|logπθ(y+)-logπref(y+)|+|logπθ(y-)-logπref(y-)|) 的 batch 均值。"
            "不是 GRPO/PPO 的 KL，禁止与 train/kl 合并。"
        ),
    },
    "train/beta": {
        "unit": "1",
        "step_kind": "optimizer_step",
        "algorithms": ["dpo"],
        "definition": "DPO/IPO 当前 β（固定或动态控制器输出）。",
    },
    "val/mean_kl": {
        "unit": "nat",
        "step_kind": "optimizer_step",
        "algorithms": ["dpo"],
        "definition": "验证集上与 train/mean_kl 同一定义的 KL 代理。",
    },
    # ----- GRPO 专属 -----
    "train/kl": {
        "unit": "nat",
        "step_kind": "optimizer_step",
        "algorithms": ["grpo"],
        "definition": (
            "GRPO 策略相对参考（或 disable_adapter）的序列级 KL 估计，见 grpo_loss。"
            "与 DPO train/mean_kl 定义不同。"
        ),
    },
    "train/reward_mean": {
        "unit": "1",
        "step_kind": "optimizer_step",
        "algorithms": ["grpo"],
        "definition": "GRPO 更新所用 batch 的奖励均值。",
    },
    "rollout/reward/mean": {
        "unit": "1",
        "step_kind": "rollout_step",
        "algorithms": ["grpo"],
        "definition": "Rollout 阶段组采样奖励均值（若引擎上报）。",
    },
    # ----- PPO 专属 -----
    "ppo/reward": {
        "unit": "1",
        "step_kind": "rollout_step",
        "algorithms": ["ppo"],
        "definition": "在线循环单条 completion 的标量奖励。",
    },
    "ppo/reward_mean_ema": {
        "unit": "1",
        "step_kind": "optimizer_step",
        "algorithms": ["ppo"],
        "definition": "在线循环累计奖励均值（截至当前 step）。",
    },
}


def catalog_keys_for(algorithm: str) -> list[str]:
    algo = str(algorithm).lower()
    return [k for k, meta in METRIC_CATALOG.items() if algo in meta.get("algorithms", [])]
