"""自研 GRPO 目标函数（token 级 PPO-clip + 组内相对优势 + KL）。

参考 DeepSeekMath GRPO：同一 prompt 采样一组回答，用组内奖励的均值/方差
做优势，再对策略做 clip 更新，并用参考模型 KL 约束偏移。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.algorithms.grpo_utils import group_relative_advantages


def compute_grpo_loss(
    policy_logps: torch.Tensor,
    old_logps: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    ref_logps: torch.Tensor | None = None,
    clip_range: float = 0.2,
    beta: float = 0.04,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Token 级 GRPO loss。

    Args:
        policy_logps: 当前策略的 token log π_θ，形状 (N, T)。
        old_logps: 采样时策略的 token log π_old（无梯度），形状 (N, T)。
        advantages: 组内相对优势，形状 (N,)；会广播到每个 completion token。
        completion_mask: 1 表示该位置是生成 token，形状 (N, T)。
        ref_logps: 参考策略 token log π_ref；为 None 或 beta=0 时不加 KL。
        clip_range: PPO clip ε。
        beta: KL 惩罚系数。

    Returns:
        (loss, metrics)。loss 越小越好（已取负号，可直接 backward）。
    """
    if policy_logps.shape != old_logps.shape or policy_logps.shape != completion_mask.shape:
        raise ValueError(
            "policy_logps / old_logps / completion_mask 形状必须一致，"
            f"得到 {tuple(policy_logps.shape)} {tuple(old_logps.shape)} {tuple(completion_mask.shape)}"
        )
    if advantages.shape[0] != policy_logps.shape[0]:
        raise ValueError("advantages 的 batch 维必须等于 N")

    mask = completion_mask.to(dtype=policy_logps.dtype)
    adv = advantages.detach().unsqueeze(-1)
    old = old_logps.detach()

    # 重要性采样比 ρ = π_θ / π_old
    log_ratio = policy_logps - old
    ratio = torch.exp(log_ratio)
    unclipped = ratio * adv
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    clipped = clipped_ratio * adv
    # 最大化目标的下界 → 训练时最小化其相反数
    per_token = -torch.min(unclipped, clipped)

    kl_mean = policy_logps.new_zeros(())
    if ref_logps is not None and beta > 0:
        # Schulman k3 估计：exp(log π_ref - log π_θ) - (log π_ref - log π_θ) - 1
        delta = ref_logps.detach() - policy_logps
        kl = torch.exp(delta) - delta - 1.0
        per_token = per_token + beta * kl
        denom_kl = mask.sum().clamp(min=1.0)
        kl_mean = (kl * mask).sum() / denom_kl

    denom = mask.sum().clamp(min=1.0)
    loss = (per_token * mask).sum() / denom
    # mask 全 0 时仍保持计算图
    loss = loss + policy_logps.sum() * 0.0

    with torch.no_grad():
        clipped_frac = ((ratio - clipped_ratio).abs() > 1e-6).to(policy_logps.dtype)
        clipfrac = (clipped_frac * mask).sum() / denom

    metrics = {
        "loss": float(loss.detach().item()),
        "policy/clipfrac": float(clipfrac.item()),
        "policy/kl": float(kl_mean.detach().item()),
        "policy/ratio_mean": float(((ratio * mask).sum() / denom).item()),
        "advantages/mean": float(advantages.detach().mean().item()),
    }
    return loss, metrics


def advantages_from_rewards(
    rewards: torch.Tensor,
    num_generations: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """由扁平奖励 (B*G,) 计算组内相对优势。"""
    return group_relative_advantages(rewards, num_generations=num_generations, eps=eps)


def sequence_logprobs_from_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """
    由 logits (N, T, V) 得到每个「下一 token」的 logprob，形状 (N, T-1)。

    token_logp[:, t] = log π(input_ids[:, t+1] | input_ids[:, :t+1])。
    """
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    return log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)


def completion_mask_from_prompt_lens(
    seq_len_minus_1: int,
    prompt_lens: torch.Tensor,
    attention_mask_target: torch.Tensor,
) -> torch.Tensor:
    """
    构造 completion mask，形状 (N, T-1)。

    prompt_lens: 每条样本的 prompt token 数（不含左 padding）。
    attention_mask_target: 与 shift 后的 label 对齐，形状 (N, T-1)，1 表示非 pad。
    """
    n = prompt_lens.shape[0]
    positions = torch.arange(seq_len_minus_1, device=prompt_lens.device).unsqueeze(0).expand(n, -1)
    # logprob 位置 t 对应 token index t+1；completion 从 prompt_len 开始
    in_completion = positions >= (prompt_lens.unsqueeze(-1) - 1).clamp(min=0)
    return (in_completion & attention_mask_target.bool()).to(dtype=torch.float32)
