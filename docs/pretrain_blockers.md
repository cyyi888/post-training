# 训练前阻断项排查

对照当前仓库代码与配置的结论（本地审查，**未**在 GPU 上实测）。

| 级别 | 含义 |
|------|------|
| **BLOCKER** | 不处理则 Smoke 大概率直接失败 |
| **RISK** | 特定路径失败或有安全/兼容隐患 |
| **NOTE** | 已知行为，按文档操作即可 |

---

## Hydra / 配置

| 项 | 级别 | 说明 |
|----|------|------|
| 默认模型 | NOTE | `configs/base.yaml` → `model=qwen2.5-0.5b`（LoRA）；Full 用 `model=qwen2.5-0.5b-full` |
| 算法与数据配对 | **BLOCKER** | DPO 必须 `data=dpo_demo`（`prompt/chosen/rejected`）；SFT/GRPO/PPO 用 `data=gsm8k`。错配会在投影/Trainer 处失败 |
| `eval.checkpoint` | **BLOCKER** | 默认 `null`；`scripts/eval.py` 必须传 `eval.checkpoint=sft\|dpo\|…` 或路径 |
| `report_to` | NOTE | 顶层可为 `wandb` 标记；HF `TrainingArguments.report_to` 由 `resolve_hf_report_to` **恒为 none**（MetricSink） |
| Pipeline `dry_run` | NOTE | 已改为默认 **`true`**；真跑需 `pipeline.dry_run=false` |
| Ablation `dry_run` | NOTE | 默认 **`true`**；正式 9-run 勿在 Smoke 开启 |
| `engineering.mixed_precision=auto` | RISK | 无 bf16 的 GPU 会回落 fp16；极老卡需显式 `no`/`fp16` |

---

## 模型加载（Qwen2.5-0.5B）

| 项 | 级别 | 说明 |
|----|------|------|
| 模型 ID | NOTE | `Qwen/Qwen2.5-0.5B-Instruct`；禁止换基座 |
| `HF_ENDPOINT` | RISK | 默认镜像 `https://hf-mirror.com`（`prepare_runtime` / `train.py` `setdefault`）；墙外可覆盖为空或官方 |
| `torch_dtype: bfloat16` | RISK | 不支持 bf16 的设备：改 `engineering.mixed_precision=fp16` 或模型 dtype |
| LoRA target | NOTE | `q/k/v/o_proj`；与 Qwen2 结构匹配 |
| 信任远程码 | NOTE | `trust_remote_code: true` |

---

## 数据字段

| 算法 | 投影字段（`src/data/projectors.py`） | 配置 |
|------|--------------------------------------|------|
| SFT | `messages`（system/user/assistant） | `data=gsm8k` |
| DPO | `prompt` / `chosen` / `rejected`（chat messages） | `data=dpo_demo` |
| GRPO/PPO | `prompt` + `ground_truth`（+ HumanEval 的 `test`/`entry_point`） | `data=gsm8k` 或 `humaneval` |

| 项 | 级别 | 说明 |
|----|------|------|
| 首次下载 | **BLOCKER** | 需能访问 HF：`openai/gsm8k`、`banghua/DL-DPO-Dataset`（及可选 `openai/openai_humaneval`） |
| `max_train_samples` | NOTE | gsm8k 默认 512；Smoke 靠 `training.max_steps` 截断即可 |

---

## TRL / PEFT 兼容

| 项 | 级别 | 说明 |
|----|------|------|
| 声明版本 | **BLOCKER**（已修） | 必须 `trl>=0.15,<1` + `transformers>=4.51,<5`。trl **1.x** 移除 `get_batch_loss_metrics`/`concatenated_forward`，自研 DPO 必崩 |
| SFT | NOTE | `trl.SFTTrainer` + 自研 sampler/grad_clip |
| DPO | NOTE | CustomDPOTrainer 依赖上述 0.x 钩子；勿升 trl 1.x |
| PPO | **BLOCKER（对照路径）** | 声明栈 `PPOTrainer` 无 classic `step()`；默认 `online_loop` 启动即失败 → **改用 GRPO** |
| PEFT resume | NOTE | `adapter_config.json` → `PeftModel.from_pretrained`；跨阶段 `init_from` 只加载权重 |

---

## Checkpoint 衔接

| 路径 | 行为 | 级别 |
|------|------|------|
| `engineering.init_from_checkpoint=sft\|dpo\|…` | 冷启动权重，步数归零；**缺权重硬失败**（禁止静默回退基座） | NOTE（Pipeline/分阶段必用） |
| `engineering.resume_from_checkpoint=true` | 同阶段续训；SFT/DPO 走 HF resume | RISK：GRPO **仅 adapter 权重** |
| 评测 `eval.checkpoint=sft` | `latest_checkpoint(outputs/checkpoints/sft)` | **BLOCKER**：需先有训出的目录 |
| Full vs LoRA | Full：`config.json`；LoRA：`adapter_config.json` + 基座 `model.name` | NOTE |

---

## 代码执行安全性

| 项 | 级别 | 说明 |
|----|------|------|
| HumanEval / `code` 奖励 | **安全 BLOCKER（裸机）** | `subprocess`+timeout+临时目录；**非**容器/seccomp。未隔离 GPU 宿主 **禁止**跑 HE/code |
| 建议 | — | Smoke **只用 GSM8K**；HumanEval 仅隔离环境、短 timeout（默认 3s） |
| 敏感词清洗 | NOTE | 默认 `filter_sensitive=false` |

---

## 快速自检命令

```bash
python -c "import torch; print('cuda', torch.cuda.is_available())"
python -c "import trl, peft, transformers; print(trl.__version__, peft.__version__, transformers.__version__)"
python -c "from src.trainers.factory import get_trainer; print(get_trainer('sft').name, get_trainer('dpo').name)"
python scripts/pipeline.py          # 应 dry-run
python scripts/ablation.py          # 应 9×dry_run，不占 GPU
```

完整上机顺序：[`gpu_smoke_test.md`](gpu_smoke_test.md)。
