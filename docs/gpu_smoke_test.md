# GPU Smoke Test 清单（真实配置）

本清单按**本仓库现有 Hydra 配置**编写，用于服务器首次上机短跑验证。  
基座固定：`Qwen/Qwen2.5-0.5B-Instruct`（默认 `model=qwen2.5-0.5b` LoRA）。

**一键脚本（推荐）**：SSH 进 GPU 机后在仓库根目录执行

```bash
bash scripts/run_gpu_bootstrap.sh
# 已装环境续跑：  bash scripts/run_gpu_bootstrap.sh --skip-install
# 从某步续跑：    bash scripts/run_gpu_bootstrap.sh --from sft_smoke
# 只做到 CPU 测： bash scripts/run_gpu_bootstrap.sh --through cpu_tests
```

失败会打印 `FAILED AT: <阶段名>` 并写入 `outputs/gpu_bootstrap_report.txt`。

**不要**在此阶段跑 `experiment.dry_run=false` 的完整 9-run 消融。  
Pipeline / 消融仅做 dry-run 或极短步数冒烟。

状态请手填：`PASS` / `FAIL` / `SKIP` / `NOT_RUN`。

---

## 0. 前置检查（阻断项）

| # | 检查项 | 命令 / 判据 | 结果 |
|---|--------|-------------|------|
| 0.1 | Python ≥3.10、CUDA 可用 | `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` → `True` | |
| 0.2 | 依赖已装 | `pip install -e ".[dev]"` | |
| 0.3 | TRL / PEFT 可导入且 **trl 为 0.x** | `python -c "import trl, peft, transformers; print(trl.__version__, peft.__version__, transformers.__version__); assert trl.__version__.startswith('0.')"` | |
| 0.4 | PPO | 声明栈下默认 **SKIP**（无 classic `step()`）；整段用 GRPO，勿因 PPO 失败阻断 Smoke | |
| 0.5 | HF 镜像 | 默认 `HF_ENDPOINT=https://hf-mirror.com`（`configs/base.yaml`）；需能拉到模型与数据集 | |
| 0.6 | 磁盘 | `outputs/` 可写；预留 ≥10GB | |
| 0.7 | W&B（可选） | `.env` 中 `WANDB_API_KEY`；无网用 `wandb=offline` | |
| 0.8 | 数据字段 | SFT/GRPO→`data=gsm8k`；DPO→`data=dpo_demo`（**勿**对 DPO 用 gsm8k） | |
| 0.9 | 代码执行安全 | HumanEval / `code` 为 subprocess+timeout，**非**强隔离；**Smoke 禁止 HE**，只用 GSM8K | |

建议环境变量（可选）：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_DISABLED=true   # 若本轮不测 W&B
```

---

## 1. SFT（最先跑）

配置：`training=sft` + `model=qwen2.5-0.5b` + `data=gsm8k`  
产物：`outputs/checkpoints/sft/`

| # | 步骤 | 命令 | 成功判据 | 结果 |
|---|------|------|----------|------|
| 1.1 | LoRA 短训 | `python scripts/train.py training=sft model=qwen2.5-0.5b data=gsm8k training.max_steps=4` | 退出码 0；目录含 adapter 或权重；无 Traceback | |
| 1.2 | 评测入口 | `python scripts/eval.py eval=gsm8k data=gsm8k eval.checkpoint=sft` | `eval_report.json`；`checkpoint_kind=lora`；有 `accuracy` | |
| 1.3 | （可选）工程开关 | `python scripts/train.py training=sft data=gsm8k engineering.mixed_precision=bf16 engineering.gradient_accumulation_steps=2 engineering.gradient_checkpointing=true training.max_steps=4` | 退出码 0 | |
| 1.4 | （可选）W&B | `python scripts/train.py training=sft data=gsm8k wandb=online training.max_steps=4` | MetricSink 曲线有 `train/*`；HF 无双写 | |

**阻断**：1.1 失败则停止后续依赖 SFT 权重的阶段。

---

## 2. DPO（接 SFT）

配置：`training=dpo` + `data=dpo_demo` + `engineering.init_from_checkpoint=sft`

| # | 步骤 | 命令 | 成功判据 | 结果 |
|---|------|------|----------|------|
| 2.1 | 冷启动短训 | `python scripts/train.py training=dpo data=dpo_demo engineering.init_from_checkpoint=sft training.max_steps=4` | 退出码 0；日志含 init_from；`outputs/checkpoints/dpo/` | |
| 2.2 | （可选）固定 β | `python scripts/train.py training=dpo data=dpo_demo training.beta_mode=fixed training.fixed_beta=0.1 engineering.init_from_checkpoint=sft training.max_steps=4` | 退出码 0 | |
| 2.3 | （可选）W&B | 同上加 `wandb=online` | 见 `train/loss`、`train/mean_kl`、`train/beta`（动态 β 时） | |

**注意**：默认 `beta_mode=dynamic`。数据必须是偏好对 `dpo_demo`。

---

## 3. GRPO

配置：`training=grpo` + `data=gsm8k`；建议 `init_from=dpo`（无 DPO 时可用 `sft`）

| # | 步骤 | 命令 | 成功判据 | 结果 |
|---|------|------|----------|------|
| 3.1 | 短训 | `python scripts/train.py training=grpo data=gsm8k engineering.init_from_checkpoint=dpo training.max_steps=2` | 退出码 0；有 rollout/train 日志；`outputs/checkpoints/grpo/` | |
| 3.2 | 评测 | `python scripts/eval.py eval=gsm8k data=gsm8k eval.checkpoint=grpo` | 报告写出；加载不崩 | |
| 3.3 | （可选）显存指标 | 3.1 开 `wandb=online` | 若有 rollout，可见 `perf/peak_mem_gb` 或等价 | |

**说明**：自研引擎，不依赖 `trl.GRPOTrainer`。`max_steps=2` 仍可能较慢（生成×组采样）。

---

## 4. PPO（对照；当前声明栈下整段 SKIP）

| # | 步骤 | 命令 | 成功判据 | 结果 |
|---|------|------|----------|------|
| 4.1 | 跳过 | — | 标 `SKIP`；主路径已用 GRPO | |

**说明**：`trl>=0.15,<1` 的 `PPOTrainer` 无 classic `step()`，默认 `online_loop` 会明确失败。不要为 Smoke 降级 trl（会破坏 DPO）。

---

## 5. Pipeline（一键链路）

配置：`configs/experiment/sft_dpo_rl.yaml`  
**默认 `pipeline.dry_run=true`**（防误训）。

| # | 步骤 | 命令 | 成功判据 | 结果 |
|---|------|------|----------|------|
| 5.1 | Dry-run | `python scripts/pipeline.py` | 打印 SFT→DPO→GRPO 命令；不占 GPU | |
| 5.2 | 真冒烟（短） | `python scripts/pipeline.py pipeline.dry_run=false pipeline.max_steps=4` | 三阶段 returncode=0；`outputs/pipeline/sft_dpo_rl/summary.yaml` | |
| 5.3 | （可选）末段 PPO | `python scripts/pipeline.py pipeline.dry_run=false pipeline.rl_algorithm=ppo pipeline.max_steps=4` | 末段 `training=ppo` | |

**不要**在 Smoke 阶段设 `pipeline.max_steps=null`（那是正式长训）。

---

## 6. 消融调度器（仅调度，不正式 9-run）

配置：`configs/experiment/ablation.yaml`  
默认：`dry_run=true`，`seeds=[12,34,56]`，`sweep.training.fixed_beta=[0.1,0.5,1.0]` → **9 planned runs**。

| # | 步骤 | 命令 | 成功判据 | 结果 |
|---|------|------|----------|------|
| 6.1 | Dry-run 规划 | `python scripts/ablation.py` | `plan.json` 含 **9** 条；各 run `status=dry_run`；**无**真实训练 | |
| 6.2 | 报告骨架 | 同 6.1 后看 `results.csv` / `report.md` | 文件存在；可含空/缺失统计 | |
| 6.3 | report_only | `python scripts/ablation.py experiment.report_only=true` | 不调用 train；可覆盖报告 | |

**禁止 Smoke 阶段**：`experiment.dry_run=false`（那是完整 GPU 消融，见 `gpu_validation.md` §7）。

---

## 建议执行顺序（复制用）

```bash
# 0) 环境
source .venv/bin/activate
python -c "import torch; assert torch.cuda.is_available()"

# 1) SFT
python scripts/train.py training=sft model=qwen2.5-0.5b data=gsm8k training.max_steps=4
python scripts/eval.py eval=gsm8k data=gsm8k eval.checkpoint=sft

# 2) DPO
python scripts/train.py training=dpo data=dpo_demo \
  engineering.init_from_checkpoint=sft training.max_steps=4

# 3) GRPO
python scripts/train.py training=grpo data=gsm8k \
  engineering.init_from_checkpoint=dpo training.max_steps=2
python scripts/eval.py eval=gsm8k data=gsm8k eval.checkpoint=grpo

# 4) PPO → SKIP（声明栈无 step()）

# 5) Pipeline
python scripts/pipeline.py
# 真冒烟（可选）：python scripts/pipeline.py pipeline.dry_run=false pipeline.max_steps=4

# 6) Ablation 调度（勿关 dry_run）
python scripts/ablation.py
```

---

## 与文档关系

| 文档 | 用途 |
|------|------|
| 本文 `gpu_smoke_test.md` | **首次上机短跑顺序**（本页） |
| `gpu_validation.md` | Week4 功能验收（MetricSink、断点、正式消融等） |
| README 工程能力矩阵 | 能力完成度总表；Smoke 通过后可将对应格改为「通过」 |
