#!/usr/bin/env bash
# PostTrainLab GPU 服务器一键上机：环境 → 下载 → CPU 测 → Smoke 短训 → dry-run。
#
# 用法（仓库根目录，已 SSH 到 GPU 机）：
#   bash scripts/run_gpu_bootstrap.sh
#   bash scripts/run_gpu_bootstrap.sh --skip-install          # 已装好 .venv
#   bash scripts/run_gpu_bootstrap.sh --from sft_smoke        # 从某步续跑
#   bash scripts/run_gpu_bootstrap.sh --through cpu_tests     # 只跑到某步
#   bash scripts/run_gpu_bootstrap.sh --with-pipeline-smoke   # 额外真跑 pipeline 短链路
#   bash scripts/run_gpu_bootstrap.sh --allow-no-cuda         # 仅调试（不会跑 GPU 训）
#
# 任一阶段非 0 退出：打印「FAILED AT: <阶段名>」并停止后续步骤。
# 成功汇总写入 outputs/gpu_bootstrap_report.txt
#
# 顺序对齐 docs/gpu_smoke_test.md；不跑 PPO / HumanEval / 完整 9-run 消融。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ---------- 可覆盖参数 ----------
PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
SFT_STEPS="${SFT_STEPS:-4}"
DPO_STEPS="${DPO_STEPS:-4}"
GRPO_STEPS="${GRPO_STEPS:-2}"
HF_ENDPOINT_DEFAULT="https://hf-mirror.com"

SKIP_INSTALL=0
SKIP_DOWNLOAD=0
SKIP_TESTS=0
SKIP_TRAIN=0
ALLOW_NO_CUDA=0
WITH_PIPELINE_SMOKE=0
FROM_STEP=""
THROUGH_STEP=""

# ---------- 阶段表（顺序即执行序）----------
# id|标题|是否需要 GPU 训练
STEPS=(
  "env_setup|创建/激活 venv 并安装依赖|0"
  "preflight|CUDA / TRL0.x / 磁盘前置检查|0"
  "download_assets|预拉 Qwen 基座 + gsm8k + dpo_demo|0"
  "cpu_tests|CPU 轻量单测（README 列表）|0"
  "sft_smoke|SFT LoRA 短训|1"
  "sft_eval|GSM8K 评测 SFT checkpoint|1"
  "dpo_smoke|DPO 短训（init_from=sft）|1"
  "grpo_smoke|GRPO 短训（init_from=dpo）|1"
  "grpo_eval|GSM8K 评测 GRPO checkpoint|1"
  "pipeline_dry|Pipeline dry-run（只打印命令）|0"
  "ablation_dry|Ablation dry-run（9-run 规划）|0"
  "pipeline_smoke|Pipeline 真短跑（可选，需 --with-pipeline-smoke）|1"
)

usage() {
  sed -n '2,16p' "$0" | sed 's/^# \?//'
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage ;;
    --skip-install) SKIP_INSTALL=1 ;;
    --skip-download) SKIP_DOWNLOAD=1 ;;
    --skip-tests) SKIP_TESTS=1 ;;
    --skip-train) SKIP_TRAIN=1 ;;
    --allow-no-cuda) ALLOW_NO_CUDA=1 ;;
    --with-pipeline-smoke) WITH_PIPELINE_SMOKE=1 ;;
    --from)
      FROM_STEP="${2:-}"; shift ;;
    --through)
      THROUGH_STEP="${2:-}"; shift ;;
    *)
      echo "未知参数: $1（--help 查看用法）" >&2
      exit 2
      ;;
  esac
  shift
done

# ---------- 日志 ----------
REPORT_DIR="$ROOT/outputs"
REPORT_FILE="$REPORT_DIR/gpu_bootstrap_report.txt"
mkdir -p "$REPORT_DIR" outputs/checkpoints outputs/logs outputs/wandb

ts() { date '+%Y-%m-%d %H:%M:%S'; }

log() { echo "[$(ts)] $*"; }
ok()  { log "PASS  $*"; echo "PASS  $*" >>"$REPORT_FILE"; }
fail() {
  local step="$1"; shift
  log "FAILED AT: ${step} — $*"
  {
    echo ""
    echo "======== FAILED AT: ${step} ========"
    echo "time: $(ts)"
    echo "detail: $*"
    echo "===================================="
  } | tee -a "$REPORT_FILE"
  exit 1
}

skip_msg() { log "SKIP  $*"; echo "SKIP  $*" >>"$REPORT_FILE"; }

{
  echo "PostTrainLab GPU bootstrap"
  echo "started: $(ts)"
  echo "root: $ROOT"
  echo "args: skip_install=$SKIP_INSTALL skip_download=$SKIP_DOWNLOAD skip_tests=$SKIP_TESTS skip_train=$SKIP_TRAIN with_pipeline_smoke=$WITH_PIPELINE_SMOKE from=${FROM_STEP:--} through=${THROUGH_STEP:--}"
  echo "----------------------------------------"
} >"$REPORT_FILE"

# ---------- 阶段门控 ----------
_seen_from=0
[[ -z "$FROM_STEP" ]] && _seen_from=1

should_run() {
  local id="$1"
  if [[ "$_seen_from" -eq 0 ]]; then
    if [[ "$id" == "$FROM_STEP" ]]; then
      _seen_from=1
    else
      return 1
    fi
  fi
  if [[ -n "$THROUGH_STEP" && "$id" == "$THROUGH_STEP" ]]; then
    # 本步仍跑；跑完后由调用方停止
    :
  fi
  return 0
}

stop_after_through() {
  local id="$1"
  if [[ -n "$THROUGH_STEP" && "$id" == "$THROUGH_STEP" ]]; then
    log "已到达 --through=${THROUGH_STEP}，停止后续阶段。"
    echo "STOPPED_AFTER ${THROUGH_STEP}" >>"$REPORT_FILE"
    exit 0
  fi
}

run_cmd() {
  local step="$1"; shift
  log ">> ($step) $*"
  if ! "$@"; then
    fail "$step" "命令失败: $*"
  fi
}

# 对 heredoc / stdin 脚本用此函数（run_cmd 会吞掉 stdin）
run_python() {
  local step="$1"
  log ">> ($step) python - <<'...'"
  if ! python -; then
    fail "$step" "Python 检查/下载脚本失败（见上方 Traceback）"
  fi
}

# ---------- 激活 Python ----------
activate_venv() {
  if [[ -f "$VENV_DIR/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    PYTHON=python
  fi
}

# =============================================================================
# 各阶段实现
# =============================================================================

step_env_setup() {
  if [[ "$SKIP_INSTALL" -eq 1 ]]; then
    activate_venv
    skip_msg "env_setup（--skip-install）；当前: $($PYTHON -c 'import sys; print(sys.executable)')"
    return 0
  fi

  if [[ ! -d "$VENV_DIR" ]]; then
    log "创建 venv: $VENV_DIR"
    run_cmd env_setup "$PYTHON" -m venv "$VENV_DIR"
  fi
  activate_venv
  run_cmd env_setup python -m pip install --upgrade pip
  run_cmd env_setup pip install -e ".[dev]"

  if [[ ! -f .env ]]; then
    if [[ -f .env.example ]]; then
      cp .env.example .env
      log "已生成 .env（请按需填 WANDB_API_KEY）；本脚本默认 wandb=disabled"
    fi
  fi
  export HF_ENDPOINT="${HF_ENDPOINT:-$HF_ENDPOINT_DEFAULT}"
  ok "env_setup | python=$(command -v python) | HF_ENDPOINT=$HF_ENDPOINT"
}

step_preflight() {
  activate_venv
  export HF_ENDPOINT="${HF_ENDPOINT:-$HF_ENDPOINT_DEFAULT}"

  run_python preflight <<'PY'
import sys
assert sys.version_info >= (3, 10), f"需要 Python>=3.10，当前 {sys.version}"
print("python", sys.version.split()[0])
PY

  if ! python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit(2)
print("gpu", torch.cuda.get_device_name(0))
PY
  then
    if [[ "$ALLOW_NO_CUDA" -eq 1 ]]; then
      skip_msg "preflight CUDA（--allow-no-cuda）；后续 GPU 训会被跳过"
    else
      fail "preflight" "CUDA 不可用。确认驱动/PyTorch GPU 版，或加 --allow-no-cuda 只做环境与 CPU 测"
    fi
  fi

  run_python preflight <<'PY'
import trl, peft, transformers
print("trl", trl.__version__, "peft", peft.__version__, "transformers", transformers.__version__)
assert trl.__version__.startswith("0."), (
    f"trl 必须为 0.x（当前 {trl.__version__}）。"
    "请 pip install -e '.[dev]'，勿升到 trl 1.x（会破坏自研 DPO）。"
)
assert transformers.__version__.startswith("4."), (
    f"transformers 须为 4.x（当前 {transformers.__version__}），与 trl<1 配对。"
)
from src.trainers.factory import get_trainer
import src.trainers.sft_trainer  # noqa: F401
import src.trainers.dpo_trainer  # noqa: F401
import src.trainers.grpo_trainer  # noqa: F401
import src.trainers.ppo_trainer  # noqa: F401
for a in ("sft", "dpo", "grpo", "ppo"):
    assert get_trainer(a).name == a
print("trainers ok:", "sft dpo grpo ppo")
print("NOTE: PPO online_loop 在声明栈下 SKIP（无 classic step()）")
PY

  run_python preflight <<'PY'
from pathlib import Path
import shutil
p = Path("outputs")
p.mkdir(parents=True, exist_ok=True)
usage = shutil.disk_usage(p)
free_gb = usage.free / (1024**3)
print(f"outputs free_gb={free_gb:.1f}")
if free_gb < 5:
    raise SystemExit(f"磁盘剩余 {free_gb:.1f}GB < 5GB，请清理后再跑")
PY

  ok "preflight"
}

step_download_assets() {
  activate_venv
  export HF_ENDPOINT="${HF_ENDPOINT:-$HF_ENDPOINT_DEFAULT}"

  if [[ "$SKIP_DOWNLOAD" -eq 1 ]]; then
    skip_msg "download_assets（--skip-download）"
    return 0
  fi

  run_python download_assets <<'PY'
import os
os.environ.setdefault("HF_ENDPOINT", os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
print("HF_ENDPOINT", os.environ.get("HF_ENDPOINT"))

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print(">> download tokenizer/model: Qwen/Qwen2.5-0.5B-Instruct")
tok = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    trust_remote_code=True,
)
print("tokenizer ok", tok.__class__.__name__, "vocab", len(tok))

dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
    torch.float16 if torch.cuda.is_available() else torch.float32
)
print(">> download model weights dtype=", dtype)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    trust_remote_code=True,
    torch_dtype=dtype,
)
n = sum(p.numel() for p in model.parameters())
print(f"model ok params={n/1e6:.1f}M")
del model

print(">> download dataset openai/gsm8k")
ds = load_dataset("openai/gsm8k", "main")
print("gsm8k splits", list(ds.keys()), "train", len(ds["train"]))

print(">> download dataset banghua/DL-DPO-Dataset")
dpo = load_dataset("banghua/DL-DPO-Dataset")
key0 = list(dpo.keys())[0]
print("dpo_demo splits", list(dpo.keys()), key0, len(dpo[key0]))

print("download_assets done")
PY
  ok "download_assets"
}

step_cpu_tests() {
  activate_venv
  if [[ "$SKIP_TESTS" -eq 1 ]]; then
    skip_msg "cpu_tests（--skip-tests）"
    return 0
  fi

  run_cmd cpu_tests python -m pytest \
    tests/test_dpo_losses.py \
    tests/test_grpo_loss.py \
    tests/test_rewards.py \
    tests/test_evaluators.py \
    tests/test_rollout.py \
    tests/test_data_pipeline.py \
    tests/test_pipeline_offline.py \
    tests/test_pipeline_chain.py \
    tests/test_sft_baseline.py \
    tests/test_trainers_strategy.py \
    tests/test_versioning.py \
    tests/test_wandb_utils.py \
    tests/test_metric_sink.py \
    tests/test_checkpoint_eval.py \
    tests/test_result_aggregator.py \
    tests/test_ablation_report_e2e.py \
    tests/test_experiment_planner_runner.py \
    -q --tb=line

  ok "cpu_tests"
}

_cuda_ok() {
  python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>/dev/null
}

maybe_skip_gpu() {
  local step="$1"
  if [[ "$SKIP_TRAIN" -eq 1 ]]; then
    skip_msg "$step（--skip-train）"
    return 0
  fi
  if ! _cuda_ok; then
    if [[ "$ALLOW_NO_CUDA" -eq 1 ]]; then
      skip_msg "$step（无 CUDA + --allow-no-cuda）"
      return 0
    fi
    fail "$step" "需要 CUDA"
  fi
  return 1  # 表示不要 skip，继续跑
}

step_sft_smoke() {
  activate_venv
  maybe_skip_gpu sft_smoke && return 0
  run_cmd sft_smoke python scripts/train.py \
    training=sft model=qwen2.5-0.5b data=gsm8k \
    training.max_steps="$SFT_STEPS" \
    wandb=disabled
  ok "sft_smoke max_steps=$SFT_STEPS"
}

step_sft_eval() {
  activate_venv
  maybe_skip_gpu sft_eval && return 0
  run_cmd sft_eval python scripts/eval.py \
    eval=gsm8k data=gsm8k eval.checkpoint=sft \
    wandb=disabled
  ok "sft_eval"
}

step_dpo_smoke() {
  activate_venv
  maybe_skip_gpu dpo_smoke && return 0
  run_cmd dpo_smoke python scripts/train.py \
    training=dpo data=dpo_demo \
    engineering.init_from_checkpoint=sft \
    training.max_steps="$DPO_STEPS" \
    wandb=disabled
  ok "dpo_smoke max_steps=$DPO_STEPS"
}

step_grpo_smoke() {
  activate_venv
  maybe_skip_gpu grpo_smoke && return 0
  # 12GB 卡上默认 4×512 生成容易 OOM，可用环境变量缩小冒烟规模
  run_cmd grpo_smoke python scripts/train.py \
    training=grpo data=gsm8k \
    engineering.init_from_checkpoint=dpo \
    training.max_steps="$GRPO_STEPS" \
    training.num_generations="${GRPO_NUM_GENERATIONS:-4}" \
    training.max_completion_length="${GRPO_MAX_COMPLETION_LENGTH:-512}" \
    training.gradient_checkpointing="${GRPO_GRADIENT_CHECKPOINTING:-false}" \
    training.rollout.require_answer_format="${GRPO_REQUIRE_ANSWER_FORMAT:-true}" \
    training.rollout.min_valid_per_group="${GRPO_MIN_VALID_PER_GROUP:-2}" \
    wandb=disabled
  ok "grpo_smoke max_steps=$GRPO_STEPS"
}

step_grpo_eval() {
  activate_venv
  maybe_skip_gpu grpo_eval && return 0
  run_cmd grpo_eval python scripts/eval.py \
    eval=gsm8k data=gsm8k eval.checkpoint=grpo \
    wandb=disabled
  ok "grpo_eval"
}

step_pipeline_dry() {
  activate_venv
  run_cmd pipeline_dry python scripts/pipeline.py
  ok "pipeline_dry"
}

step_ablation_dry() {
  activate_venv
  run_cmd ablation_dry python scripts/ablation.py
  ok "ablation_dry"
}

step_pipeline_smoke() {
  activate_venv
  if [[ "$WITH_PIPELINE_SMOKE" -ne 1 ]]; then
    skip_msg "pipeline_smoke（未传 --with-pipeline-smoke；避免重复长训）"
    return 0
  fi
  maybe_skip_gpu pipeline_smoke && return 0
  run_cmd pipeline_smoke python scripts/pipeline.py \
    pipeline.dry_run=false \
    pipeline.max_steps="$SFT_STEPS"
  ok "pipeline_smoke"
}

# =============================================================================
# 主循环
# =============================================================================

log "======= PostTrainLab GPU bootstrap 开始 ======="
log "报告文件: $REPORT_FILE"

for entry in "${STEPS[@]}"; do
  IFS='|' read -r sid stitle _need_gpu <<<"$entry"
  if ! should_run "$sid"; then
    continue
  fi
  log "----- STEP: $sid | $stitle -----"
  case "$sid" in
    env_setup) step_env_setup ;;
    preflight) step_preflight ;;
    download_assets) step_download_assets ;;
    cpu_tests) step_cpu_tests ;;
    sft_smoke) step_sft_smoke ;;
    sft_eval) step_sft_eval ;;
    dpo_smoke) step_dpo_smoke ;;
    grpo_smoke) step_grpo_smoke ;;
    grpo_eval) step_grpo_eval ;;
    pipeline_dry) step_pipeline_dry ;;
    ablation_dry) step_ablation_dry ;;
    pipeline_smoke) step_pipeline_smoke ;;
    *) fail "$sid" "未知阶段（脚本内部错误）" ;;
  esac
  stop_after_through "$sid"
done

{
  echo "----------------------------------------"
  echo "ALL STEPS FINISHED OK"
  echo "ended: $(ts)"
} | tee -a "$REPORT_FILE"

log "======= 全部完成 ======="
log "汇总: $REPORT_FILE"
log "下一步正式长训示例（手动）："
log "  python scripts/train.py training=sft data=gsm8k training.max_steps=null"
log "  python scripts/pipeline.py pipeline.dry_run=false pipeline.max_steps=null"
log "完整消融（耗时长，确认后再跑）："
log "  python scripts/ablation.py experiment.dry_run=false"
