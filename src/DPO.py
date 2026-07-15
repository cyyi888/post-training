import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import pandas as pd
from datasets import load_dataset,Dataset
from transformers import TrainingArguments, AutoTokenizer,AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from helper import generate_response,test_model_with_questions,load_model_and_tokenizer

USE_GPU = True

questions = [
    "Give me an 1-sentence introduction of LLM.",
    "Calculate 1+1-1",
    "What is the difference between thread and a process?"]

#train_dataset = load_dataset("banghua/DL-SFT-Dataset")["train"]
#display_dataset(train_dataset)

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model, tokenizer = load_model_and_tokenizer(model_name, use_gpu=USE_GPU)

#加dpo前
test_model_with_questions(model, tokenizer, questions, title="instruct Model (before DPO) Output")

dpo_trainer_trainer = DPOTrainer(
    model=model,
    args=dpo_config,
    train_dataset=train_dataset,
    processing_class=tokenizer,
)
dpo_trainer.train()

#加sft后
test_model_with_questions(model, tokenizer, questions, title="DPO Model Output")

del model, tokenizer

