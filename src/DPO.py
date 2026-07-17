import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import pandas as pd
from datasets import load_dataset,Dataset
from transformers import TrainingArguments, AutoTokenizer,AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from helper import generate_response,test_model_with_questions,load_model_and_tokenizer

POS_NAME = "Deep Qwen"
ORG_NAME = "Qwen"
SYSTEM_PROMPT = "You are a helpful assistant."

USE_GPU = True

questions = [
    "What is your name?",
    "Are you chatGPT?",
    "Tell me about your name and organization."]

def build_dpo_chatml(example):
    msgs = example["conversations"]
    prompt = next(m["value"]for m in reversed(msgs) if m["from"] == "human")
    try:
        rejected_resp = generate_response(model, tokenizer, prompt)
    except Exception as e:
        rejected_resp = "ERROR:failed to generate response"
        print(f"Generation error for prompt:{prompt}\n {e}")
    chosen_resp = rejected_resp.replace(ORG_NAME, POS_NAME)
    chosen = [
        {"role":"system", "content":SYSTEM_PROMPT},
        {"role":"user", "content":prompt},
        {"role":"assistant", "content":chosen_resp},
    ]
    rejected = [
        {"role":"system", "content":SYSTEM_PROMPT},
        {"role":"user", "content":prompt},
        {"role":"assistant", "content":rejected_resp},
    ]

    return {"chosen":chosen, "rejected":rejected}

raw_ds = load_dataset("mrfakename/identity",split="train")

#show the first 5 elements of the raw dataset
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 0)

sample_df = raw_ds.select(range(5)).to_pandas()
display_dataset(sample_df)

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model, tokenizer = load_model_and_tokenizer(model_name, use_gpu=USE_GPU)

#加DPO前
test_model_with_questions(model, tokenizer, questions, title="instruct Model (before DPO) Output")

dpo_trainer_trainer = DPOTrainer(
    model=model,
    args=dpo_config,
    train_dataset=train_dataset,
    processing_class=tokenizer,
)
dpo_trainer.train()

#加DPO后
test_model_with_questions(model, tokenizer, questions, title="DPO Model Output")

del model, tokenizer

