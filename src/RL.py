import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,TrainingArguments
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset,Dataset
from helper import generate_response,test_model_with_questions,load_model_and_tokenizer
import re
import pandas as pd
from tqdm import tqdm

SYSTEM_PROMPT = (
    "You are a helpful assistant that solves problems step by step."
    "Always include the final numeric answer inside \\boxed{}."
)

def reward_func(completions,ground_truth,**kwargs):
    #用正则表达式捕获方框内的内容
    matches = [re.search(r'\\boxed{(.*?)}', completion[0]['content']) for completion in completions]
    content = [match.group(1) if match else " " for match in matches]
    #回答正确奖励1，否则0
    return [1.0 if c == gt else 0.0 for c, gt in zip(content, ground_truth)]

sample_pred = [[{"role":"assistant",
            "content": r"...calculate the answer is \\boxed{71}."}]]
ground_truth = ["72"]
reward = reward_func(sample_pred, ground_truth)
print(f"Negative Sample Reward: {reward}")


