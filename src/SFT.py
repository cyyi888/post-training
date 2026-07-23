import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import pandas as pd
from datasets import load_dataset,Dataset
from transformers import TrainingArguments, AutoTokenizer,AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from helper import generate_response,test_model_with_questions,load_model_and_tokenizer,display_dataset

sft_config = SFTConfig(
    output_dir="checkpoints/smollm2-135m-sft",
    learning_rate=8e-5,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=False,
    logging_steps=2,
    max_length=512,
    assistant_only_loss=True,
    report_to="none",
)

USE_GPU = True

questions = [
    "Give me an 1-sentence introduction of LLM.",
    "Calculate 1+1-1",
    "What is the difference between thread and a process?"]

train_dataset = load_dataset("banghua/DL-SFT-Dataset")["train"]
display_dataset(train_dataset)

model_name = "HuggingFaceTB/SmolLM2-135M"
model, tokenizer = load_model_and_tokenizer(model_name, use_gpu=USE_GPU)

#加sft前
test_model_with_questions(model, tokenizer, questions, title="Base Model (before SFT) Output")

sft_trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    processing_class=tokenizer,
)
sft_trainer.train()

#加sft后
test_model_with_questions(model, tokenizer, questions, title="SFT Model Output")
