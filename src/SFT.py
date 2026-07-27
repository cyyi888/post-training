import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from helper import test_model_with_questions, load_model_and_tokenizer, display_dataset

MAX_EVAL_SAMPLES = 32
USE_GPU = True

sft_config = SFTConfig(
    output_dir="checkpoints/smollm2-135m-sft",
    learning_rate=8e-5,
    num_train_epochs=2,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=False,
    logging_steps=2,
    eval_strategy="epoch",
    max_length=512,
    assistant_only_loss=True,
    report_to="none",
)

questions = [
    "Give me an 1-sentence introduction of LLM.",
    "Calculate 1+1-1",
    "What is the difference between thread and a process?",
]

full_dataset = load_dataset("banghua/DL-SFT-Dataset")["train"]
display_dataset(full_dataset)

n_eval = min(MAX_EVAL_SAMPLES, max(1, len(full_dataset) // 10))
split = full_dataset.train_test_split(test_size=n_eval, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]
print(f"train samples: {len(train_dataset)}, eval samples: {len(eval_dataset)}")

model_name = "HuggingFaceTB/SmolLM2-135M"
model, tokenizer = load_model_and_tokenizer(model_name, use_gpu=USE_GPU)

# 加 SFT 前
test_model_with_questions(model, tokenizer, questions, title="Base Model (before SFT) Output")

sft_trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)
sft_trainer.train()

# eval：验证集 loss + 定性问答
model.eval()
eval_metrics = sft_trainer.evaluate()
print("\n=== Eval metrics ===")
print(eval_metrics)

test_model_with_questions(model, tokenizer, questions, title="SFT Model Output")
