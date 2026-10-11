# GPU 服务器验收清单（Week 4）

本文记录**未来 GPU 环境**需执行的验收项。本地 CPU 阶段仅完成 `IMPLEMENTED` + `CPU_VERIFIED`（Mock / 单元测试），**不编造**真实训练或评测数值。

状态约定：

| 标记 | 含义 |
|------|------|
| `IMPLEMENTED` | 生产功能代码已落地 |
| `CPU_VERIFIED` | 本地 Mock / 单测已通过 |
| `GPU_PENDING` | 需在 GPU 服务器用真实 Qwen2.5-0.5B 验证 |
| `NOT_RUN` | 当前环境未执行（缺依赖 / 无 GPU） |

基座硬性约定：只用 `Qwen/Qwen2.5-0.5B-Instruct`（Full 或 LoRA）。

---

## 1. SFT / DPO MetricSink 真实 WandB

| 字段 | 内容 |
|------|------|
| 状态 | `GPU_PENDING` |
| 预期命令 | `python scripts/train.py training=sft data=gsm8k wandb=online training.max_steps=20` |
| | `python scripts/train.py training=dpo data=dpo_demo wandb=online training.max_steps=20` |
| 成功判据 | W&B 出现 `train/*`（及 SFT/DPO 过程指标）；进程无异常退出 |
| 风险 | 未配置 `WANDB_API_KEY`；offline 模式只写本地 `outputs/wandb/` |

## 2. Full Fine-tuning Checkpoint 加载与评测

| 字段 | 内容 |
|------|------|
| 状态 | `IMPLEMENTED` + `CPU_VERIFIED`（Mock 路径）+ `GPU_PENDING` |
| 预期命令 | `python scripts/train.py training=sft model=qwen2.5-0.5b-full training.max_steps=20` |
| | `python scripts/eval.py eval=gsm8k eval.checkpoint=sft` |
| 成功判据 | `eval_report.json` 含 `checkpoint_kind=full` 与数值指标；加载无报错 |
| 风险 | 全量权重显存不足；checkpoint 目录缺 `config.json` |

## 3. LoRA Adapter Checkpoint 加载与评测

| 字段 | 内容 |
|------|------|
| 状态 | `IMPLEMENTED` + `CPU_VERIFIED` + `GPU_PENDING` |
| 预期命令 | `python scripts/train.py training=sft model=qwen2.5-0.5b training.max_steps=20` |
| | `python scripts/eval.py eval=gsm8k eval.checkpoint=sft` |
| 成功判据 | `checkpoint_kind=lora`；基座名与 adapter 匹配；指标写出 |
| 风险 | 基座与 adapter 不匹配；仅有 adapter 未设 `model.name` |

## 4. GRPO / PPO Checkpoint 兼容性

| 字段 | 内容 |
|------|------|
| 状态 | `GPU_PENDING` |
| 预期命令 | `python scripts/train.py training=grpo data=gsm8k training.max_steps=5` |
| | `python scripts/eval.py eval=gsm8k eval.checkpoint=grpo` |
| | `python scripts/train.py training=ppo data=gsm8k training.max_steps=5` |
| | `python scripts/eval.py eval=gsm8k eval.checkpoint=ppo` |
| 成功判据 | 评测入口能识别 LoRA/Full；不因保存格式崩溃 |
| 风险 | GRPO 只存最终目录、无 HF `checkpoint-*`；PPO online 路径格式差异 |

## 5. GSM8K 真实模型评测

| 字段 | 内容 |
|------|------|
| 状态 | `GPU_PENDING` |
| 预期命令 | `python scripts/eval.py eval=gsm8k data=gsm8k eval.checkpoint=<path>` |
| 成功判据 | `accuracy` / `format_rate` 为有限浮点；报告 `validation_status` 可人工改为已验收 |
| 风险 | 数据集下载；生成慢；勿在 CI 无 GPU 时强跑 |

## 6. HumanEval 真实生成与安全执行

| 字段 | 内容 |
|------|------|
| 状态 | `GPU_PENDING` |
| 预期命令 | `python scripts/eval.py eval=humaneval data=humaneval eval.checkpoint=<path> eval.num_samples=1` |
| 成功判据 | `pass_at_1` 写出；沙箱 timeout 生效、无主机破坏 |
| 风险 | 代码执行安全；超时过短导致假阴性 |

## 7. DPO Beta 消融真实 GPU 训练

| 字段 | 内容 |
|------|------|
| 状态 | Planner/Runner `IMPLEMENTED` + `CPU_VERIFIED`；真实训练 `GPU_PENDING` |
| 预期命令 | `python scripts/ablation.py experiment.dry_run=false experiment.max_retries=1` |
| 成功判据 | 9 个 run（3 β × 3 seed）均有 checkpoint + `eval_report.json`；`results.csv` / `summary.csv` / `report.md` 生成 |
| 风险 | 默认 `dry_run=true` 防误训；务必显式关闭；总 GPU 时间长 |

## 8. 中断、失败恢复与重复运行

| 字段 | 内容 |
|------|------|
| 状态 | Mock `CPU_VERIFIED`；真实 `GPU_PENDING` |
| 预期命令 | 跑消融中途 Ctrl+C → 再执行同一命令；故意让某 run 失败后重跑 |
| 成功判据 | `succeeded` 被 skip；`failed` 可重试；目录与 run_id 稳定 |
| 风险 | 半写入 checkpoint；需确认 `skip_completed` 行为符合预期 |

## 9. 自动报告引用真实评测结果

| 字段 | 内容 |
|------|------|
| 状态 | Fixture `CPU_VERIFIED`（见 `tests/test_ablation_report_e2e.py`）；真实 `GPU_PENDING` |
| 预期命令 | 完成第 7 项后：`python scripts/ablation.py experiment.report_only=true` |
| | 检查 `outputs/ablation/ablation_dpo_beta/{results,summary}.csv`、`report.md`、`plots/*.png` |
| 成功判据 | 均值/样本标准差与手工对若干 `eval_report.json` 一致；**无** `data_source=test_fixture`；MD 引用 PNG |
| 风险 | 误把本地 Fixture（如 `outputs/ablation_demo_cpu_fixture/`）当作真实结论 |

### 本地 CPU 报告闭环（已验收，非 GPU）

```bash
py -3.12 -m pytest tests/test_result_aggregator.py tests/test_ablation_report_e2e.py -q
```

样例产物（**test_fixture**）：`outputs/ablation_demo_cpu_fixture/ablation/ablation_dpo_beta_cpu_fixture/`

---

## 推荐 GPU 冒烟顺序

```bash
# 0) 环境
pip install -e ".[dev]"
# 可选：pip install matplotlib  # 消融对比图

# 1) 短训 + 评测（LoRA）
python scripts/train.py training=sft data=gsm8k training.max_steps=20
python scripts/eval.py eval=gsm8k eval.checkpoint=sft

# 2) 消融 dry-run（应立刻结束，仅写 plan）
python scripts/ablation.py

# 3) 真实消融（耗时）
python scripts/ablation.py experiment.dry_run=false
```

## 本地 CPU 已跑通的测试（勿与 GPU 验收混淆）

```bash
py -3.12 -m pytest tests/test_checkpoint_eval.py \
  tests/test_experiment_planner_runner.py \
  tests/test_result_aggregator.py -q
```
