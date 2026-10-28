"""GRPO 可插拔奖励函数引擎。

内置：
- accuracy / format / length：通用答案与格式
- steps：数学题逐步过程分（非单纯 0/1）
- code：代码编译 + 测例执行（HumanEval 风格）
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable

RewardFn = Callable[..., Any]

_REGISTRY: dict[str, RewardFn] = {}

_BOXED_RE = re.compile(r"\\boxed\{(.*?)\}")
_HASH_RE = re.compile(r"####\s*([^\n]+)")
_STEP_MARKER_RE = re.compile(
    r"(?im)^(?:\s*(?:step\s*\d+|步骤\s*\d+|第[一二三四五六七八九十\d]+步|[0-9]+[\.\)、]))"
)
_EQUATION_RE = re.compile(
    r"(?:=|\\times|\\div|\+|\-|×|÷|/|\*)\s*-?\d+(?:\.\d+)?|-?\d+(?:\.\d+)?\s*(?:=|\\times|\\div|\+|×|÷)"
)
_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*([\s\S]*?)```", re.IGNORECASE)


def register_reward(name: str):
    def deco(fn: RewardFn):
        _REGISTRY[name] = fn
        return fn

    return deco


def available_rewards() -> list[str]:
    return sorted(_REGISTRY)


def _extract_text(completion: Any) -> str:
    if isinstance(completion, list):
        return completion[0]["content"]
    return str(completion)


def extract_answer(text: str) -> str:
    """优先 ``\\boxed{...}``，其次 GSM8K ``#### 答案``。"""
    boxed = _BOXED_RE.search(text)
    if boxed:
        return boxed.group(1).strip()
    hashed = _HASH_RE.search(text)
    if hashed:
        return hashed.group(1).strip()
    return ""


def has_answer_format(text: str) -> bool:
    """是否包含标准答案标记（boxed 或 ####）。"""
    return bool(_BOXED_RE.search(text) or _HASH_RE.search(text))


def extract_python_code(text: str) -> str:
    """从模型输出中提取 Python 代码（优先 markdown 代码块）。

    注意：不可对整段 ``.strip()``，否则会削掉函数体缩进导致语法错误。
    """
    fences = _CODE_FENCE_RE.findall(text)
    if fences:
        # fence 内仍保留缩进，只去掉首尾空行
        parts = []
        for f in fences:
            lines = f.splitlines()
            while lines and not lines[0].strip():
                lines.pop(0)
            while lines and not lines[-1].strip():
                lines.pop()
            if lines:
                parts.append("\n".join(lines))
        return "\n\n".join(parts)
    # 无 fence 时：去掉常见说明前缀，保留疑似代码（含缩进）
    lines = text.splitlines()
    code_lines: list[str] = []
    started = False
    for line in lines:
        if not started and (
            line.startswith(("def ", "class ", "import ", "from ", "    ", "\t"))
            or line.strip().startswith("@")
        ):
            started = True
        if started:
            code_lines.append(line)
    if code_lines:
        while code_lines and not code_lines[-1].strip():
            code_lines.pop()
        return "\n".join(code_lines)
    return text.rstrip()


def score_math_steps(text: str, ground_truth: Any | None = None) -> float:
    """
    数学逐步打分，输出 ``[0, 1]``。

    组成：
    - 结构分（最多 0.4）：多行推理、Step/步骤编号
    - 演算分（最多 0.3）：中间等式 / 数值推导痕迹
    - 收尾分（最多 0.3）：写出最终答案标记；若提供 gt 且答对再加满
    """
    cleaned = text.strip()
    if not cleaned:
        return 0.0

    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    n_lines = len(lines)
    step_hits = len(_STEP_MARKER_RE.findall(cleaned))
    # 结构：鼓励多步展开
    structure = 0.0
    if n_lines >= 2:
        structure += 0.15
    if n_lines >= 4:
        structure += 0.10
    if step_hits >= 1:
        structure += 0.10
    if step_hits >= 3:
        structure += 0.05
    structure = min(0.4, structure)

    # 演算：等式、运算符夹数字
    eq_hits = len(_EQUATION_RE.findall(cleaned))
    digit_lines = sum(1 for ln in lines if re.search(r"\d", ln))
    calc = 0.0
    if eq_hits >= 1 or digit_lines >= 2:
        calc += 0.15
    if eq_hits >= 2 or digit_lines >= 3:
        calc += 0.10
    if eq_hits >= 3:
        calc += 0.05
    calc = min(0.3, calc)

    # 收尾：答案格式 + 正确性
    finale = 0.0
    pred = extract_answer(cleaned)
    if pred:
        finale += 0.15
    if ground_truth is not None and pred and pred == str(ground_truth).strip():
        finale += 0.15
    elif ground_truth is not None and pred:
        finale += 0.05  # 有答案但错：少量过程收尾分
    finale = min(0.3, finale)

    return float(min(1.0, structure + calc + finale))


def run_humaneval_tests(
    prompt: str,
    completion: str,
    test: str,
    entry_point: str,
    *,
    timeout: float = 3.0,
) -> tuple[bool, str]:
    """
    在子进程中执行 HumanEval 风格：prompt+completion+test，再调用 check(entry_point)。

    返回 (通过?, 原因)。
    """
    return _run_humaneval_once(
        prompt, completion, test, entry_point, timeout=timeout
    )


def _run_humaneval_once(
    prompt: str,
    completion: str,
    test: str,
    entry_point: str,
    *,
    timeout: float = 3.0,
) -> tuple[bool, str]:
    """
    在子进程中执行 HumanEval 风格：prompt+completion+test，再调用 check(entry_point)。

    返回 (通过?, 原因)。
    """
    code_body = extract_python_code(completion)
    # HumanEval：prompt 已含函数头，completion 多为函数体；若 completion 已含完整 def 则直接用
    if re.search(rf"(?m)^def\s+{re.escape(entry_point)}\s*\(", code_body):
        program_src = code_body
    else:
        program_src = (prompt or "") + code_body
    # 语法检查（主进程，快速失败）
    try:
        ast.parse(program_src)
    except SyntaxError as exc:
        return False, f"syntax:{exc.msg}"

    check_call = f"\n\n{test}\n\ncheck({entry_point})\n"
    full = program_src + check_call
    with tempfile.TemporaryDirectory(prefix="ptl_code_reward_") as tmp:
        path = Path(tmp) / "solution.py"
        path.write_text(full, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(path)],
                capture_output=True,
                text=True,
                timeout=float(timeout),
                cwd=tmp,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
        if proc.returncode == 0:
            return True, "pass"
        err = (proc.stderr or proc.stdout or "fail").strip().splitlines()
        tail = err[-1] if err else "fail"
        return False, tail[:200]


def score_code_execution(
    completion: str,
    *,
    prompt: str = "",
    test: str | None = None,
    entry_point: str | None = None,
    timeout: float = 3.0,
) -> float:
    """
    代码奖励 ``[0, 1]``：
    - 无 test/entry_point：仅语法可解析 → 0.2，否则 0
    - 可编译但测例失败 → 0.3
    - 测例全过 → 1.0
    """
    code = extract_python_code(completion)
    if not code.strip():
        return 0.0
    if not test or not entry_point:
        try:
            ast.parse(code if re.search(r"(?m)^def\s+", code) else (prompt or "") + code)
            return 0.2
        except SyntaxError:
            return 0.0
    ok, reason = _run_humaneval_once(
        prompt or "",
        completion,
        str(test),
        str(entry_point),
        timeout=timeout,
    )
    if ok:
        return 1.0
    if reason.startswith("syntax"):
        return 0.0
    if reason == "timeout":
        return 0.1
    # 运行起来但断言失败等
    return 0.3


@register_reward("accuracy")
def reward_accuracy(completions: Sequence[Any], ground_truth: Sequence[str], **_: Any) -> list[float]:
    """最终答案与 ground_truth 精确匹配（boxed / ####）。"""
    scores = []
    for completion, gt in zip(completions, ground_truth):
        pred = extract_answer(_extract_text(completion))
        scores.append(1.0 if pred == str(gt).strip() else 0.0)
    return scores


@register_reward("format")
def reward_format(completions: Sequence[Any], **_: Any) -> list[float]:
    """有标准答案标记（boxed 或 ####）得 1，否则 0。"""
    scores = []
    for completion in completions:
        text = _extract_text(completion)
        scores.append(1.0 if has_answer_format(text) else 0.0)
    return scores


@register_reward("length")
def reward_length(
    completions: Sequence[Any],
    *,
    completion_lengths: Sequence[int] | None = None,
    length_reward: dict[str, Any] | None = None,
    **_: Any,
) -> list[float]:
    """
    按 completion token 数分层打分，输出 ``[0, 1]``。

    - 短于 ``min_tokens`` 或长于 ``max_tokens`` → 0
    - 落在 ``[target_min, target_max]`` → 1
    - 介于边界与目标区间之间 → 线性插值
    """
    cfg = dict(length_reward or {})
    min_tokens = int(cfg.get("min_tokens", 16))
    max_tokens = int(cfg.get("max_tokens", 256))
    target_min = int(cfg.get("target_min", 32))
    target_max = int(cfg.get("target_max", 192))
    if min_tokens > target_min or target_min > target_max or target_max > max_tokens:
        raise ValueError(
            "length_reward 需满足 min_tokens <= target_min <= target_max <= max_tokens"
        )

    scores: list[float] = []
    for i, completion in enumerate(completions):
        if completion_lengths is not None and i < len(completion_lengths):
            n = int(completion_lengths[i])
        else:
            text = _extract_text(completion).strip()
            n = len(text.split()) if text else 0
        if n < min_tokens or n > max_tokens:
            scores.append(0.0)
        elif target_min <= n <= target_max:
            scores.append(1.0)
        elif n < target_min:
            denom = max(target_min - min_tokens, 1)
            scores.append(max(0.0, min(1.0, (n - min_tokens) / denom)))
        else:
            denom = max(max_tokens - target_max, 1)
            scores.append(max(0.0, min(1.0, (max_tokens - n) / denom)))
    return scores


@register_reward("steps")
def reward_steps(
    completions: Sequence[Any],
    ground_truth: Sequence[Any] | None = None,
    **_: Any,
) -> list[float]:
    """数学题逐步过程分（结构 + 演算 + 收尾），非二元 0/1。"""
    scores: list[float] = []
    for i, completion in enumerate(completions):
        gt = None
        if ground_truth is not None and i < len(ground_truth):
            gt = ground_truth[i]
        scores.append(score_math_steps(_extract_text(completion), gt))
    return scores


@register_reward("code")
def reward_code(
    completions: Sequence[Any],
    *,
    prompts: Sequence[str] | None = None,
    tests: Sequence[Any] | None = None,
    entry_points: Sequence[Any] | None = None,
    code_reward: dict[str, Any] | None = None,
    **_: Any,
) -> list[float]:
    """
    代码奖励：语法检查 + HumanEval 测例执行。

    需要 Rollout 传入 ``prompts`` / ``tests`` / ``entry_points``（与 completions 对齐）。
    """
    cfg = dict(code_reward or {})
    timeout = float(cfg.get("timeout", 3.0))
    scores: list[float] = []
    for i, completion in enumerate(completions):
        prompt = ""
        test = None
        entry = None
        if prompts is not None and i < len(prompts):
            prompt = str(prompts[i] or "")
        if tests is not None and i < len(tests):
            test = tests[i]
        if entry_points is not None and i < len(entry_points):
            entry = entry_points[i]
        scores.append(
            score_code_execution(
                _extract_text(completion),
                prompt=prompt,
                test=test,
                entry_point=entry,
                timeout=timeout,
            )
        )
    return scores


class RewardEngine:
    """Combine multiple named reward functions with weights."""

    def __init__(
        self,
        names: Sequence[str],
        weights: Sequence[float] | None = None,
        *,
        length_reward: dict[str, Any] | None = None,
        code_reward: dict[str, Any] | None = None,
    ):
        missing = [n for n in names if n not in _REGISTRY]
        if missing:
            raise KeyError(f"Unknown reward funcs: {missing}. Available: {list(_REGISTRY)}")
        self.names = list(names)
        if weights is None:
            self.weights = [1.0] * len(self.names)
        else:
            if len(weights) != len(names):
                raise ValueError("weights length must match funcs")
            self.weights = list(weights)
        self.length_reward = dict(length_reward or {})
        self.code_reward = dict(code_reward or {})

    def score_detailed(
        self,
        completions: Sequence[Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """返回总分与分项分数，供 Rollout 日志使用。"""
        n = len(completions)
        if not self.names:
            return {
                "total": [0.0] * n,
                "parts": {},
            }
        call_kwargs = dict(kwargs)
        if self.length_reward and "length_reward" not in call_kwargs:
            call_kwargs["length_reward"] = self.length_reward
        if self.code_reward and "code_reward" not in call_kwargs:
            call_kwargs["code_reward"] = self.code_reward

        parts: dict[str, list[float]] = {}
        total = [0.0] * n
        for name, w in zip(self.names, self.weights):
            partial = [float(v) for v in _REGISTRY[name](completions, **call_kwargs)]
            if len(partial) != n:
                raise RuntimeError(f"奖励 {name} 返回 {len(partial)} 条，期望 {n}")
            parts[name] = partial
            for i, v in enumerate(partial):
                total[i] += w * v
        alias = {
            "accuracy": "reward_correct",
            "format": "reward_format",
            "length": "reward_length",
            "steps": "reward_steps",
            "code": "reward_code",
        }
        for src, dst in alias.items():
            if src in parts and dst not in parts:
                parts[dst] = list(parts[src])
        return {"total": total, "parts": parts, "reward_total": total}

    def __call__(self, completions: Sequence[Any], **kwargs: Any) -> list[float]:
        return list(self.score_detailed(completions, **kwargs)["total"])


def build_reward_funcs(cfg: Any) -> RewardFn:
    """Build a single callable from Hydra `training.reward` config."""
    names = list(cfg.funcs)
    weights = list(cfg.get("weights", [1.0] * len(names)))
    length_reward = None
    code_reward = None
    if hasattr(cfg, "get"):
        length_reward = cfg.get("length_reward")
        code_reward = cfg.get("code_reward")
    else:
        length_reward = getattr(cfg, "length_reward", None)
        code_reward = getattr(cfg, "code_reward", None)

    def _to_dict(obj: Any) -> dict[str, Any] | None:
        if obj is None:
            return None
        try:
            from omegaconf import OmegaConf

            return dict(OmegaConf.to_container(obj, resolve=True))  # type: ignore[arg-type]
        except Exception:
            return dict(obj)

    return RewardEngine(
        names,
        weights,
        length_reward=_to_dict(length_reward),
        code_reward=_to_dict(code_reward),
    )
