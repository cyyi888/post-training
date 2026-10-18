# PostTrainLab 训练准入报告（Pre-Training Readiness）

审查日期：2026-10-15  
环境：本地 Windows CPU · Python 3.12.1 · **无 GPU**  
基座硬性约定：`Qwen/Qwen2.5-0.5B-Instruct`（仅 Full / LoRA）

---

## 最终判定

| 项 | 结论 |
|----|------|
| **CPU 工程验收** | **PASS** |
| **GPU Smoke Test 准备程度** | **READY_FOR_GPU_SMOKE**（主路径 SFT→DPO→GRPO） |
| **正式训练准入** | **PENDING_GPU_VALIDATION** |

说明：未做真实 GPU 训练/评测，故正式训练不得标 GO。PPO 对照路径在声明依赖下 **不可作为 Smoke 阻断解除条件**（见问题清单）。

---

## 8.1 工程验收表

| 检查项目 | 状态 | 证据 | 问题 |
|----------|------|------|------|
| Hydra 配置 | **PASS**（静态 compose） | 组合 SFT LoRA/Full、DPO 动/定 β、IPO、GRPO/HE、PPO、eval、pipeline/ablation 均可解析；模型名均为 Qwen2.5-0.5B | — |
| SFT 入口 | **PASS**（静态+CPU） | `scripts/train.py`→`SFTStageTrainer`→TRL `SFTTrainer`；`max_steps`/sampler/grad_clip/MetricSink Callback 接线有测 | 真实训 GPU_PENDING |
| DPO/IPO 入口 | **PASS**（静态+CPU；依赖已钉死） | CustomDPOTrainer 覆盖 `get_batch_loss_metrics`；需 `trl<1` | 曾被 trl 1.15 破坏，已 pin |
| GRPO 入口 | **PASS**（静态+CPU） | 自研 Rollout→`GRPOEngine`；不依赖 `trl.GRPOTrainer` | 真实 generate/更新 GPU_PENDING |
| PPO 入口 | **FAIL***（入口在，默认路径不可跑） | `trl` 0.15–0.x 无 classic `step()`；`online_loop=true` 启动即报错 | 不阻断 GRPO Smoke |
| Checkpoint | **PASS**（CPU Mock） | init_from / resume / eval 三语义；缺权重硬失败；`test_checkpoint_eval` | 完整续训 GPU_PENDING |
| MetricSink | **PASS**（CPU） | `test_metric_sink`；HF `report_to=none` | 真实 W&B online GPU_PENDING |
| Eval | **PASS**（CPU Mock） | `evaluate(config, checkpoint)`；缺 ckpt 报错 | 真实生成 GPU_PENDING |
| Pipeline | **PASS**（dry-run 实测） | 默认 `dry_run=true`；打印 SFT→DPO→GRPO 命令 | 真跑 GPU_PENDING |
| Ablation | **PASS**（dry-run 实测） | 9 runs dry_run；plan/csv/md | 真训 GPU_PENDING |
| Report | **PASS**（CPU fixture） | Aggregator + e2e fixture | 真实数值 GPU_PENDING |
| README 一致性 | **PASS**（已修订） | BaseTrainer 口径、依赖钉死、PPO 限制、矩阵已对齐代码 | — |

\* PPO：生产代码与注册存在，但对齐声明依赖后默认执行路径失败 → 验收表记 FAIL，**不**把整体 CPU 验收打成 FAIL。

---

## 8.2 问题清单

### P0（已修复）

1. **TRL 1.x 破坏自研 DPO**  
   - 位置：`pyproject.toml` 原 `trl>=0.15` 无上界；pip 解析到 1.15 后 `DPOTrainer` 无 `get_batch_loss_metrics` / `concatenated_forward`。  
   - 影响：GPU 上 DPO/IPO 训练必崩。  
   - 修复：钉死 `transformers>=4.51,<5` + `trl>=0.15,<1`。本地验证：trl **0.19.1** + transformers **4.57.6**，钩子恢复。

2. **`init_from_checkpoint` 静默回退基座**  
   - 位置：`src/core/base.py` `resolve_init_from` / `load_init_checkpoint`。  
   - 影响：阶段别名无权重时继续训基座，污染跨阶段结论。  
   - 修复：显式别名/路径缺失或无法识别时 **raise**，禁止静默回退。

### P0 / 安全（未改代码行为；文档升级）

3. **HumanEval / code 奖励执行隔离不足**  
   - 位置：`src/algorithms/reward_functions.py` `_run_humaneval_once`（`subprocess` + 临时目录 + timeout，无容器/seccomp）。  
   - 影响：不可信生成代码可在宿主执行。  
   - 状态：**安全 BLOCKER（仅 HumanEval/code 路径）**。GPU Smoke **只用 GSM8K**；HumanEval 须隔离环境，见 `gpu_smoke_test.md`。

### P1（记录，不阻断最小 Smoke）

4. **PPO 默认 online_loop 与声明栈不兼容**  
   - 位置：`src/trainers/ppo_trainer.py`；trl 0.19 `PPOTrainer` 有 `train()` 无 `step()`。  
   - 影响：`training=ppo` 默认启动失败；已加启动期明确报错。  
   - 处置：Smoke/正式主路径用 **GRPO**；不降级 trl 以换 PPO（会冲击 DPO）。

5. **GRPO/PPO 完整断点恢复仅为权重**  
   - 已在 README 矩阵标注「部分」。

6. **`engineering.mixed_precision=auto`**  
   - 无 bf16 的卡回落 fp16；极老卡需显式 `no`/`fp16`。

### P2

7. SFT/DPO 显存监控未接线。  
8. DDP / 完整 resume / 真实 W&B online 未 GPU 实测。

---

## 8.3 能力矩阵

以 README「工程能力测试矩阵」为准（已更新 PPO 列与依赖说明）。**无任何 GPU_VERIFIED。**

---

## 8.4 测试结果（真实执行）

```text
# 解释器
C:\Users\...\Python312\python.exe  (3.12.1)

# 安装后钉死栈
trl 0.19.1  transformers 4.57.6  peft 0.21.2  torch 2.14.1+cpu

# CPU 轻量（README 列表）
python -m pytest tests/test_dpo_losses.py tests/test_grpo_loss.py tests/test_rewards.py \
  tests/test_evaluators.py tests/test_rollout.py tests/test_data_pipeline.py \
  tests/test_pipeline_offline.py tests/test_pipeline_chain.py tests/test_sft_baseline.py \
  tests/test_trainers_strategy.py tests/test_versioning.py tests/test_wandb_utils.py \
  tests/test_metric_sink.py tests/test_checkpoint_eval.py \
  tests/test_result_aggregator.py tests/test_ablation_report_e2e.py \
  tests/test_experiment_planner_runner.py -q
→ **133 passed**（0 failed / 0 skipped）

# GPU 流水线（无 CUDA）
python -m pytest tests/test_dpo_pipeline.py tests/test_grpo_pipeline.py tests/test_ppo_pipeline.py -v
→ **12 skipped**

# Dry-run CLI
python scripts/pipeline.py   → dry_run，SFT→DPO→GRPO 命令正确
python scripts/ablation.py   → 9×dry_run，写出 plan/csv/md
```

未运行：真实模型下载、真实训练、真实评测、真实消融。

---

## 8.5 GPU 验收清单

见 [`gpu_smoke_test.md`](gpu_smoke_test.md)（最短上机顺序）与 [`gpu_validation.md`](gpu_validation.md)（Week4 功能项）。  
服务器安装务必：`pip install -e ".[dev]"`，确认 `trl.__version__` 为 **0.x**（`<1`）。

---

## 8.6 准入结论（复述）

- CPU 工程验收：**PASS**
- GPU Smoke：**READY_FOR_GPU_SMOKE**（SFT / DPO / GRPO / Pipeline dry-run / Ablation dry-run）
- 正式训练：**PENDING_GPU_VALIDATION**
- PPO / HumanEval：分别作为对照路径与安全受限路径，**不**纳入首轮 Smoke 通过条件
