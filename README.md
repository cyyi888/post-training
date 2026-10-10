# PostTrainLab

统一的大模型后训练实验平台：SFT → DPO/IPO → GRPO，配合 Hydra 分层配置与消融实验调度。

## 硬性约定：模型只用 Qwen2.5-0.5B

**本仓库基座模型固定为 `Qwen/Qwen2.5-0.5B-Instruct`（配置见 `configs/model/qwen2.5-0.5b.yaml` / `qwen2.5-0.5b-full.yaml`）。**

- 训练、评测、消融、集成/冒烟测试 **一律只用该模型**（Full 或 LoRA 变体均可，仍是同一权重族）。
- **禁止**为省事引入 GPT-2、tiny-*、其它随机小模型或第三方基座作为替代。
- 改代码、补测试、写脚本、改文档时都要遵守；需要更小算力时用 `max_steps` / 截断长度 / LoRA，而不是换模型。

## 架构

```
PostTrainLab/
├── configs/          # Hydra 分层配置（model / data / training / eval / experiment）
├── src/
│   ├── core/         # 统一基类与接口
│   ├── data/         # 统一 Sample 流水线：适配器/清洗/划分/版本/Chat Template
│   ├── models/       # 模型加载、LoRA
│   ├── algorithms/   # 自研 DPO/IPO Loss、动态 β、GRPO、SFT 采样/裁剪
│   ├── trainers/     # SFT(TRL) / DPO(TRL+自研Loss) / GRPO / PPO(TRL 基线)
│   ├── evaluators/   # 任务评测与报告
│   ├── experiment/   # 消融调度与追踪
│   └── utils/        # 通用工具
├── scripts/          # 一键训练 / 评测 / 消融
├── tests/            # 单元测试（含 DPO Loss 正确性）
├── benchmarks/       # 评测数据集
└── outputs/          # checkpoints / logs / wandb
```

```mermaid
flowchart LR
  Raw[原始数据] --> Adapter[Format Adapter]
  Adapter --> Sample[统一 Sample]
  Sample --> Clean[清洗]
  Clean --> Split[划分]
  Split --> Ver[版本哈希]
  Ver --> Proj[投影 SFT/DPO/GRPO]
  Proj --> Trainer
  ChatTpl[Chat Template] --> Trainer
  ChatTpl --> Eval[evaluators]
  SFT[SFT] -->|init_from| DPO[DPO]
  DPO -->|init_from| RL[GRPO/PPO]
```

## 数据流水线

一份原始数据经统一 `Sample` 后可投影到任意算法：

```bash
# 同一 gsm8k 数据 → GRPO
python scripts/train.py training=grpo data=gsm8k

# 偏好数据 → DPO
python scripts/train.py training=dpo data=dpo_demo

# 启用敏感词过滤 / 自定义划分
python scripts/train.py data.cleaning.filter_sensitive=true data.split.ratios=[0.8,0.1,0.1]
```

数据版本写入 `outputs/data_versions/<name>/<version>/dataset_version.json`，
同目录下有 `latest.json` 指针；并写入本次实验的 `cfg.data_version`。

复现实验时可强制校验：

```bash
python scripts/train.py data.expected_version=v-toy data.expected_hash=abc123def456
```

## 快速开始

### 环境

```bash
# 本地
pip install -e ".[dev]"

# 外部服务器（推荐）
bash scripts/setup_server.sh
source .venv/bin/activate
# 编辑 .env，填入 WANDB_API_KEY（从 https://wandb.ai/authorize 复制）
# 之后一般不必再执行 wandb login

# 或 Docker
docker build -t posttrainlab .
docker run --gpus all -it -v $(pwd)/outputs:/workspace/outputs posttrainlab
```

`.env.example` 可复制为 `.env`；进程启动时会自动加载（密钥勿提交 git）。

### 训练

日志、W&B、断点续训、分布式，以及梯度累积、混合精度、梯度检查点、随机种子和学习率调度都由 `BaseTrainer` 统一处理。SFT / DPO / GRPO / PPO 只实现算法。各算法 yaml 里的学习率、累积步数、调度器仍然生效；要全局覆盖时用 `engineering.*=...`。

#### SFT 基线（TRL + Full/LoRA 对比）

基于 `trl.SFTTrainer`；同一权重 `Qwen2.5-0.5B-Instruct`，用两套预设切换微调方式。自研：`sampler`（随机 / 长度分组 / 分桶）与 `grad_clip`（norm / value / adaptive）。

```bash
# LoRA
python scripts/train.py training=sft model=qwen2.5-0.5b data=gsm8k training.max_steps=20

# Full（对照）
python scripts/train.py training=sft model=qwen2.5-0.5b-full data=gsm8k training.max_steps=20

# 切换采样 / 裁剪策略
python scripts/train.py training=sft model=qwen2.5-0.5b \
  training.sampler.mode=bucket training.grad_clip.strategy=norm
```

#### DPO / IPO（TRL + 自研 Loss / 动态 β）

基于 `trl.DPOTrainer` 扩展；自研 `dpo_losses`（DPO / IPO 合页）与 `DynamicBetaController`。配置一键切换：

```bash
# 标准 DPO + 动态 β
python scripts/train.py training=dpo data=dpo_demo wandb=online

# IPO + 固定 β
python scripts/train.py training=dpo data=dpo_demo \
  training.loss_type=ipo training.beta_mode=fixed training.fixed_beta=0.1

# DPO + 固定 β
python scripts/train.py training=dpo data=dpo_demo \
  training.loss_type=dpo training.beta_mode=fixed training.fixed_beta=0.5
```

WandB 关注曲线：`train/loss`、`train/mean_kl`、`train/beta`（以及 `val/loss`、`val/mean_kl`）。

离线流水线测试（**需要 GPU**；无 CUDA 会自动 skip）：

```bash
pytest tests/test_dpo_pipeline.py -v
# 仅跑 GPU 冒烟
pytest tests/test_dpo_pipeline.py::test_dpo_pipeline_trl_train_smoke_gpu -v
```

Loss / 动态 β 纯逻辑单测（CPU 即可）：

```bash
pytest tests/test_dpo_losses.py -q
```

#### GRPO（自研完整 Rollout）与 PPO（TRL 对照）

GRPO **不调用** `trl.GRPOTrainer`。在线采样走自研 6 步管线（[`src/algorithms/rollout.py`](src/algorithms/rollout.py)）：

1. **固定档分桶 + 动态 batch**（默认 128/256/512；短档大 batch、长档小 batch）
2. 批量 `generate`（`no_grad` + `eval`；Hydra 配置 `temperature` / `top_p` / `top_k` / `num_generations`；批后清 KV）
3. 后处理：解码清洗；答案支持 `\boxed{}` 与 GSM8K `####`；过短/过长丢弃；槽位拒绝采样 + **有效不足半组时整组重采**
4. 分层奖励：正确性 + 格式 + 长度 + **逐步过程分** + **代码测例**（[`reward_functions.py`](src/algorithms/reward_functions.py)）
5. 组内相对优势归一化，组装 `input_ids` / mask / reward / advantage（另附 `prompt_ids` / `answer_ids`）
6. 指标：奖励 mean/max/min/std、合格率、tok/s、batch 耗时、峰值显存 → WandB（`rollout/*`）

随后由 [`GRPOEngine`](src/algorithms/grpo_engine.py) 做 clip + KL 更新。

PPO 与 GRPO **共用** `RewardEngine`；默认 `ppo.online_loop=true` 走 generate → 奖励 → `PPOTrainer.step`（保证 `ground_truth` / 测例元数据可用）。完整 6 步 Rollout + 组内优势仍以 GRPO 为主路径。

```bash
# GRPO 主路径（GSM8K：含 steps 逐步奖励）
python scripts/train.py training=grpo data=gsm8k training.max_steps=20

# HumanEval / 代码题（code 编译+测例奖励）
python scripts/train.py training=grpo_humaneval data=humaneval training.max_steps=20

# 关闭分桶 / 调整拒绝采样与核采样
python scripts/train.py training=grpo data=gsm8k \
  training.rollout.bucket=false training.rollout.max_resample=1 \
  training.rollout.top_p=0.9 training.rollout.top_k=50

# 消融对比时固定生成种子
python scripts/train.py training=grpo data=gsm8k \
  training.rollout.fix_generation_seed=true training.rollout.generation_seed=42

# PPO 对照基线（需环境含 TRL PPOTrainer；可接上游 DPO 权重）
python scripts/train.py training=ppo data=gsm8k training.max_steps=20 \
  engineering.init_from_checkpoint=dpo
```

#### SFT → DPO → RL 完整链路

跨阶段用 `engineering.init_from_checkpoint` **只加载权重、步数从 0 开始**（与同阶段 `resume_from_checkpoint` 续训区分）。别名 `sft` / `dpo` / `grpo` / `ppo` 指向 `outputs/checkpoints/<stage>` 下最新目录。

```bash
# 一键链路（默认 max_steps=20 冒烟；末段 rl_algorithm=grpo）
python scripts/pipeline.py
python scripts/pipeline.py pipeline.dry_run=true          # 只打印命令
python scripts/pipeline.py pipeline.rl_algorithm=ppo      # 末段改 PPO
python scripts/pipeline.py pipeline.max_steps=null        # 按各 yaml 正式训

# 或分阶段手动衔接
python scripts/train.py training=sft data=gsm8k training.max_steps=20
python scripts/train.py training=dpo data=dpo_demo \
  engineering.init_from_checkpoint=sft training.max_steps=20
python scripts/train.py training=grpo data=gsm8k \
  engineering.init_from_checkpoint=dpo training.max_steps=20
```

### 测试分层

- **CPU（轻量，无 CUDA 也应全绿）**：公式 / 奖励 / Rollout 控制流 / 数据流水线 / 策略注册等。
- **GPU（无 CUDA 自动 skip；基座仅 Qwen2.5-0.5B）**：DPO / GRPO / PPO 端到端短训冒烟。

```bash
# CPU 轻量（日常）
pytest tests/test_dpo_losses.py tests/test_grpo_loss.py tests/test_rewards.py \
  tests/test_rollout.py tests/test_data_pipeline.py tests/test_pipeline_offline.py \
  tests/test_pipeline_chain.py tests/test_sft_baseline.py tests/test_trainers_strategy.py \
  tests/test_versioning.py tests/test_wandb_utils.py -q

# GPU 流水线冒烟
pytest tests/test_dpo_pipeline.py tests/test_grpo_pipeline.py tests/test_ppo_pipeline.py -v
```

```bash
python scripts/train.py training=sft data=gsm8k
python scripts/train.py training=dpo data=dpo_demo
python scripts/train.py training=grpo data=gsm8k

# 开启 WandB
python scripts/train.py training=dpo wandb=online
python scripts/train.py training=grpo wandb=offline

# 从最新 checkpoint 接着训
python scripts/train.py training=sft engineering.resume_from_checkpoint=true

# 多卡
torchrun --nproc_per_node=2 scripts/train.py training=grpo data=gsm8k
```

### 日志与产物

- Hydra 每次运行目录：`outputs/logs/{algorithm}/{timestamp}/`
  - `run.log` — 控制台同级文件日志
  - `resolved_config.yaml` — 解析后的完整配置
- W&B 本地目录：`outputs/wandb/`（由 `configs/wandb/*.yaml` 控制）

```bash
# 覆盖日志级别
python scripts/train.py logging.level=DEBUG

# 自定义 run 名 / 项目
python scripts/train.py wandb=online wandb.project=my-ptl run_name=dpo-beta01
```

### 评测

```bash
python scripts/eval.py eval=gsm8k model=qwen2.5-0.5b
```

### 消融实验

```bash
python scripts/ablation.py experiment=ablation
```

## 核心算法亮点

| 模块 | 说明 |
|------|------|
| `algorithms/dpo_losses.py` + `dynamic_beta.py` | 自研 DPO / IPO（公式+Config+兼容入口）+ 动态 β；`dpo_loss` / `dpo_variants` 为再导出；训练见 `trainers/dpo_trainer` |
| `algorithms/reward_functions.py` | GRPO 可插拔奖励：accuracy / format / length / **steps（逐步）** / **code（编译+测例）** |
| `algorithms/rollout.py` | GRPO 6 步 Rollout：分桶、批量生成、拒绝采样、组内组装、指标 |
| `algorithms/grpo_loss.py` + `grpo_engine.py` | 自研 GRPO：组内优势、PPO-clip、参考 KL；引擎消费 AssembledBatch |
| `trainers/ppo_trainer.py` | TRL `PPOTrainer` + 自研奖励在线循环；与 GRPO 对照消融 |
| `scripts/pipeline.py` | SFT→DPO→RL 一键链路（`init_from_checkpoint` 衔接） |

## 实验结果

训练与评测产物统一写入 `outputs/`：

- `outputs/checkpoints/` — 模型权重
- `outputs/logs/` — 训练日志
- `outputs/wandb/` — W&B 本地缓存

（具体数值表在完成首轮实验后回填至此。）
