# GPU 服务器验收清单（Week 4）

本文记录**未来 GPU 环境**需执行的验收项。本地 CPU 阶段仅完成 `IMPLEMENTED` + `CPU_VERIFIED`（Mock / 单元测试），**不编造**真实训练或评测数值。

完整准入结论见 [`training_readiness.md`](training_readiness.md)。上机短跑顺序见 [`gpu_smoke_test.md`](gpu_smoke_test.md)。

状态约定：

| 标记 | 含义 |
|------|------|
| `IMPLEMENTED` | 生产功能代码已落地 |
| `CPU_VERIFIED` | 本地 Mock / 单测已通过 |
| `GPU_PENDING` | 需在 GPU 服务器用真实 Qwen2.5-0.5B 验证 |
| `NOT_RUN` | 当前环境未执行（缺依赖 / 无 GPU） |
| `BLOCKED` | 当前声明依赖或安全策略下不可跑 |

基座硬性约定：只用 `Qwen/Qwen2.5-0.5B-Instruct`（Full 或 LoRA）。

**依赖钉死（必读）**：`transformers>=4.51,<5`，`trl>=0.15,<1`。服务器上确认：

```bash
python -c "import trl, transformers; print(trl.__version__, transformers.__version__); assert trl.__version__.startswith('0.')"
```

---

## 0. 环境（最短）

| 检查 | 命令 / 判据 | 成功条件 |
|------|-------------|----------|
| CUDA | `python -c "import torch; assert torch.cuda.is_available()"` | True |
| 依赖 | `pip install -e ".[dev]"` | trl 为 0.x |
| 模型可达 | HF / `HF_ENDPOINT` 可拉 Qwen2.5-0.5B | 无鉴权死循环 |
| 数据 | gsm8k + dpo_demo | 可下载或本地缓存 |
| 磁盘 | `outputs/` | ≥10GB |
| W&B | 可选；无网用 `wandb=offline` 或 `WANDB_DISABLED=true` | 不挡训练 |

---

## 1. SFT / DPO MetricSink 真实 WandB

| 字段 | 内容 |
|------|------|
| 状态 | `GPU_PENDING` |
| 预期命令 | `python scripts/train.py training=sft data=gsm8k wandb=online training.max_steps=4` |
| | `python scripts/train.py training=dpo data=dpo_demo engineering.init_from_checkpoint=sft wandb=online training.max_steps=4` |
| 成功判据 | W&B 出现 `train/*`；HF 无双写；进程退出码 0 |
| 风险 | 未配置 `WANDB_API_KEY`；offline 只写 `outputs/wandb/` |

## 2. Full Fine-tuning Checkpoint 加载与评测

| 字段 | 内容 |
|------|------|
| 状态 | `IMPLEMENTED` + `CPU_VERIFIED` + `GPU_PENDING` |
| 预期命令 | `python scripts/train.py training=sft model=qwen2.5-0.5b-full data=gsm8k training.max_steps=4` |
| | `python scripts/eval.py eval=gsm8k data=gsm8k eval.checkpoint=sft` |
| 成功判据 | `eval_report.json` 含 `checkpoint_kind=full`；加载无报错 |
| 风险 | 全量显存不足 |

## 3. LoRA Adapter Checkpoint 加载与评测

| 字段 | 内容 |
|------|------|
| 状态 | `IMPLEMENTED` + `CPU_VERIFIED` + `GPU_PENDING` |
| 预期命令 | `python scripts/train.py training=sft model=qwen2.5-0.5b data=gsm8k training.max_steps=4` |
| | `python scripts/eval.py eval=gsm8k data=gsm8k eval.checkpoint=sft` |
| 成功判据 | `checkpoint_kind=lora`；指标写出 |
| 风险 | 基座与 adapter 不匹配 |

## 4. GRPO Checkpoint 兼容性

| 字段 | 内容 |
|------|------|
| 状态 | `GPU_PENDING` |
| 预期命令 | `python scripts/train.py training=grpo data=gsm8k engineering.init_from_checkpoint=dpo training.max_steps=2` |
| | `python scripts/eval.py eval=gsm8k data=gsm8k eval.checkpoint=grpo` |
| 成功判据 | 评测可加载；rollout 指标有限值；无 NaN |
| 风险 | GRPO 常存最终目录、无 HF `checkpoint-*`；空有效样本 |

## 5. PPO（对照；当前 BLOCKED）

| 字段 | 内容 |
|------|------|
| 状态 | **`BLOCKED`**（声明栈无 classic `PPOTrainer.step`） |
| 预期命令 | 不适用；请用 GRPO |
| 成功判据 | — |
| 风险 | 强行降级旧 trl 会破坏 DPO |

## 6. GSM8K 真实模型评测

| 字段 | 内容 |
|------|------|
| 状态 | `GPU_PENDING` |
| 预期命令 | `python scripts/eval.py eval=gsm8k data=gsm8k eval.checkpoint=sft` |
| 成功判据 | `accuracy` / `format_rate` 为有限浮点 |
| 风险 | 数据集下载；生成慢 |

## 7. HumanEval（安全受限）

| 字段 | 内容 |
|------|------|
| 状态 | **安全 `BLOCKED`（未隔离宿主）** + 隔离环境后 `GPU_PENDING` |
| 预期命令 | 仅在容器/沙箱：`python scripts/eval.py eval=humaneval ...` |
| 成功判据 | timeout 生效；无主机破坏 |
| 风险 | 现实现为 `subprocess`+timeout，**非**强隔离；首轮 Smoke **禁止**在裸机跑 |

## 8. DPO Beta 消融真实 GPU 训练

| 字段 | 内容 |
|------|------|
| 状态 | Planner/Runner `CPU_VERIFIED`；真实训练 `GPU_PENDING` |
| 预期命令 | 先 `python scripts/ablation.py`（dry_run）；通过后再 `experiment.dry_run=false` |
| 成功判据 | 9 run 均有 checkpoint + eval 报告；CSV/MD/PNG |
| 风险 | 默认 `dry_run=true`；总 GPU 时间长 |

## 9. 中断、失败恢复与重复运行

| 字段 | 内容 |
|------|------|
| 状态 | Mock `CPU_VERIFIED`；真实 `GPU_PENDING` |
| 成功判据 | `succeeded` skip；`failed` 可重试；run_id 稳定 |

## 10. 自动报告引用真实评测结果

| 字段 | 内容 |
|------|------|
| 状态 | Fixture `CPU_VERIFIED`；真实 `GPU_PENDING` |
| 预期命令 | `python scripts/ablation.py experiment.report_only=true` |
| 成功判据 | **无** `data_source=test_fixture`；MD 引用 PNG |

### 本地 CPU 报告闭环（已验收，非 GPU）

```bash
py -3.12 -m pytest tests/test_result_aggregator.py tests/test_ablation_report_e2e.py -q
```

---

## 推荐 GPU 冒烟顺序（最短）

```bash
pip install -e ".[dev]"
python -c "import torch, trl; assert torch.cuda.is_available(); assert trl.__version__.startswith('0.')"

python scripts/train.py training=sft model=qwen2.5-0.5b data=gsm8k training.max_steps=4
python scripts/eval.py eval=gsm8k data=gsm8k eval.checkpoint=sft

python scripts/train.py training=dpo data=dpo_demo \
  engineering.init_from_checkpoint=sft training.max_steps=4

python scripts/train.py training=grpo data=gsm8k \
  engineering.init_from_checkpoint=dpo training.max_steps=2
python scripts/eval.py eval=gsm8k data=gsm8k eval.checkpoint=grpo

python scripts/pipeline.py
python scripts/ablation.py
```

成功条件摘要：各短训退出码 0；checkpoint 可被 eval 加载；loss/reward 有限；MetricSink 无双写；**不**跑 PPO / HumanEval / 完整 9-run 消融。
