#!/usr/bin/env bash
# 在外部服务器上一键创建 Python 环境并安装依赖。
# 用法（在仓库根目录）：
#   bash scripts/setup_server.sh
#   source .venv/bin/activate
#   # 编辑 .env 填入 WANDB_API_KEY 后即可训练

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

echo "==> 使用解释器: $($PYTHON --version 2>&1)"
echo "==> 创建虚拟环境: $VENV_DIR"
$PYTHON -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "==> 升级 pip"
pip install --upgrade pip

echo "==> 安装 PostTrainLab（含 wandb 等依赖）"
pip install -e ".[dev]"

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "==> 已生成 .env，请编辑填入 WANDB_API_KEY："
  echo "    nano .env   # 或 vim .env"
else
  echo "==> 已存在 .env，跳过复制"
fi

mkdir -p outputs/checkpoints outputs/logs outputs/wandb

cat <<'EOF'

==> 环境就绪。后续每次登录服务器：

  cd /path/to/post-training
  source .venv/bin/activate

  # 确认 .env 里已有 WANDB_API_KEY 后：
  python scripts/train.py training=dpo wandb=online

说明：
  - pip install 已包含 wandb，一般不必再单独 pip install wandb
  - 有 WANDB_API_KEY 时无需交互 wandb login
  - 网络访问不了 wandb.ai 时用：wandb=offline
EOF
