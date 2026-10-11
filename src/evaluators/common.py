"""任务评测共享工具。"""

from __future__ import annotations

from typing import Any, Mapping

from omegaconf import DictConfig, OmegaConf

from src.utils.generation import generate_response


def eval_section(cfg: Any) -> dict[str, Any]:
    """读取 Hydra ``cfg.eval``；若无嵌套则把根配置当作评测段。"""
    if cfg is None:
        return {}
    if isinstance(cfg, DictConfig):
        container = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(container, dict):
            return {}
        nested = container.get("eval")
        if isinstance(nested, dict):
            return dict(nested)
        # 已传入 eval 子配置本身
        if "task" in container or "max_new_tokens" in container:
            return dict(container)
        return {}
    if isinstance(cfg, Mapping):
        nested = cfg.get("eval")
        if isinstance(nested, Mapping):
            return dict(nested)
        return dict(cfg)
    return {}


def user_and_system(example: Mapping[str, Any]) -> tuple[str, str | None]:
    """从 GRPO 投影样本取出 user / system 文本。"""
    prompt = example.get("prompt")
    if isinstance(prompt, list):
        user_q = next(m["content"] for m in prompt if m.get("role") == "user")
        system = next(
            (m["content"] for m in prompt if m.get("role") == "system"),
            None,
        )
        return str(user_q), system
    return str(prompt or ""), None


def generate_completion(
    model: Any,
    tokenizer: Any,
    example: Mapping[str, Any],
    *,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> str:
    user_q, system = user_and_system(example)
    return generate_response(
        model,
        tokenizer,
        user_q,
        system_message=system,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )


def pass_at_k(n: int, c: int, k: int) -> float:
    """HumanEval 无偏估计：``1 - C(n-c, k) / C(n, k)``。"""
    if n <= 0 or k <= 0:
        return 0.0
    if c < 0:
        c = 0
    if c > n:
        c = n
    if n - c < k:
        return 1.0
    # 逐步乘避免阶乘溢出：prod_{i=0}^{k-1} (n-c-i)/(n-i)
    prod = 1.0
    for i in range(k):
        prod *= (n - c - i) / (n - i)
    return float(1.0 - prod)
