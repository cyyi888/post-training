import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import pandas as pd
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
from helper import test_model_with_questions, load_model_and_tokenizer

# ---- 小规模 DPO 配置：优先保证能跑通 ----
dpo_config = DPOConfig(
    output_dir="checkpoints/qwen2.5-0.5b-dpo-mini",
    beta=0.1,
    # 5e-5 会崩；5e-7 几乎学不动；取中间值
    learning_rate=5e-6,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    logging_steps=10,
    max_length=512,
    report_to="none",
    save_strategy="no",
    gradient_checkpointing=False,
)

MAX_TRAIN_SAMPLES = 256
USE_GPU = True

# 偏好目标：chosen 更偏向 POS_NAME，rejected 保持 ORG_NAME
POS_NAME = "Deep Qwen"
ORG_NAME = "Qwen"

questions = [
    "What is your name?",
    "Are you ChatGPT?",
    "Tell me about your name and organization.",
]

# 1) 加载 instruct 模型
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model, tokenizer = load_model_and_tokenizer(model_name, use_gpu=USE_GPU)

# 2) DPO 前：看原始回答
test_model_with_questions(
    model,
    tokenizer,
    questions,
    title="Instruct Model (before DPO) Output",
)

# 3) 加载小规模偏好数据（数据集中 chosen 为 POS_NAME，rejected 为 ORG_NAME）
dpo_ds = load_dataset("banghua/DL-DPO-Dataset", split="train")
dpo_ds = dpo_ds.select(range(min(MAX_TRAIN_SAMPLES, len(dpo_ds))))

pd.set_option("display.max_colwidth", 120)
pd.set_option("display.width", 0)
rows = []
for i in range(min(3, len(dpo_ds))):
    ex = dpo_ds[i]
    user = next(m["content"] for m in ex["chosen"] if m["role"] == "user")
    chosen = next(m["content"] for m in ex["chosen"] if m["role"] == "assistant")
    rejected = next(m["content"] for m in ex["rejected"] if m["role"] == "assistant")
    rows.append({"user": user, "chosen": chosen[:120], "rejected": rejected[:120]})
print("\n=== DPO dataset preview ===")
print(f"POS_NAME={POS_NAME}, ORG_NAME={ORG_NAME}")
print(pd.DataFrame(rows))
print(f"train samples: {len(dpo_ds)}")

# 4) 训练 DPO
dpo_trainer = DPOTrainer(
    model=model,
    args=dpo_config,
    train_dataset=dpo_ds,
    processing_class=tokenizer,
)
dpo_trainer.train()

# 5) DPO 后：用刚训完的模型再测同一组问题，对比效果
test_model_with_questions(
    dpo_trainer.model,
    tokenizer,
    questions,
    title="DPO Model (after DPO) Output",
)

del model, tokenizer
