import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import re
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from helper import generate_response, load_model_and_tokenizer

# ============================================================
# 1) 设定参数（小规模：优先跑通）
# ============================================================
MAX_TRAIN_SAMPLES = 256
MAX_EVAL_SAMPLES = 50
USE_GPU = True
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

SYSTEM_PROMPT = (
    "You are a helpful assistant that solves problems step by step. "
    "Always include the final numeric answer inside \\boxed{}."
)

grpo_config = GRPOConfig(
    output_dir="checkpoints/qwen2.5-0.5b-grpo-mini",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=8,          # 在线采样条数；太大很慢
    num_train_epochs=2,
    max_completion_length=512,
    learning_rate=1e-5,
    logging_steps=5,
    report_to="none",
    save_strategy="no",
)


# ============================================================
# 2) 处理数据集
# ============================================================
def reward_func(completions, ground_truth, **kwargs):
    """从 \\boxed{} 提取答案，正确得 1，错误得 0。"""
    contents = []
    for completion in completions:
        text = completion[0]["content"] if isinstance(completion, list) else completion
        match = re.search(r"\\boxed\{(.*?)\}", text)
        contents.append(match.group(1).strip() if match else "")
    return [1.0 if c == gt else 0.0 for c, gt in zip(contents, ground_truth)]


def post_processing(example):
    match = re.search(r"####\s*(-?\d+)", example["answer"])
    example["ground_truth"] = match.group(1) if match else None
    example["prompt"] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["question"]},
    ]
    return example


raw = load_dataset("openai/gsm8k", "main")
train_dataset = (
    raw["train"]
    .select(range(MAX_TRAIN_SAMPLES))
    .map(post_processing)
    .remove_columns(["question", "answer"])
)
eval_dataset = (
    raw["test"]
    .select(range(MAX_EVAL_SAMPLES))
    .map(post_processing)
    .remove_columns(["question", "answer"])
)

print("\n=== train sample ===")
print(train_dataset[0])
print(f"train samples: {len(train_dataset)}, eval samples: {len(eval_dataset)}")

# 奖励函数小测
sample_pred = [[{"role": "assistant", "content": r"The answer is \boxed{71}."}]]
print("Negative sample reward:", reward_func(sample_pred, ["72"]))


# ============================================================
# 3) 加载模型 + train 模式训练（GRPO 在线强化学习）
# ============================================================
model, tokenizer = load_model_and_tokenizer(MODEL_NAME, use_gpu=USE_GPU)

grpo_trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    train_dataset=train_dataset,
    reward_funcs=reward_func,
    processing_class=tokenizer,
)
grpo_trainer.train()


# ============================================================
# 4) eval 验证
# ============================================================
model = grpo_trainer.model
model.eval()

all_preds = []
all_labels = []
print("\n=== Eval details ===")
for example in tqdm(eval_dataset, desc="eval"):
    user_question = example["prompt"][1]["content"]
    ground_truth = example["ground_truth"]
    response = generate_response(
        model,
        tokenizer,
        user_question,
        system_message=SYSTEM_PROMPT,
        max_new_tokens=512,
    )
    all_preds.append([{"role": "assistant", "content": response}])
    all_labels.append(ground_truth)
    print(f"\nQ: {user_question}")
    print(f"Pred: {response}")
    print(f"GT: {ground_truth}")


# ============================================================
# 5) 打印结果
# ============================================================
rewards = reward_func(all_preds, all_labels)
accuracy = sum(rewards) / len(rewards) if rewards else 0.0
print("\n=== GRPO Eval Result ===")
print(f"rewards: {rewards}")
print(f"Evaluation Accuracy: {accuracy:.2%}")
