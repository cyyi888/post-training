import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import pandas as pd
from datasets import load_dataset,Dataset
from transformers import TrainingArguments, AutoTokenizer,AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

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

def generate_response(model, tokenizer, user_message,system_message=None, max_new_tokens=100):
    #接下来会用 tokenizer 的 chat template 把对话格式化成模型期望的 prompt 字符串
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})

    #假设是单轮对话：只有用户一问，模型一答，没有多轮历史。
    messages.append({"role": "user", "content": user_message})

    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    
    model_input = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **model_input,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    input_len = model_input["input_ids"].shape[1]
    generated_ids = outputs[0][input_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response


def test_model_with_questions(model, tokenizer, questions, system_message=None, title="Model Output"):
    print(f"\n==={title}===")
    for i, question in enumerate(questions, 1):
        response = generate_response(model, tokenizer, question, system_message)
        print(f"\nModel Input {i}:\n{question}\nModel Output {i}:\n{response}")


def load_model_and_tokenizer(model_name, use_gpu=True):
    #load base model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if use_gpu and torch.cuda.is_available():
        model.to("cuda")
    elif use_gpu:
        print("Warning: CUDA unavailable, using CPU.")

    if not tokenizer.chat_template:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "System: {{ message['content'] }}\n"
            "{% elif message['role'] == 'user' %}"
            "User: {{ message['content'] }}\n"
            "{% elif message['role'] == 'assistant' %}"
            "Assistant: {% generation %}{{ message['content'] }}\n{% endgeneration %}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}Assistant: {% endif %}"
        )

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def display_dataset(dataset):
    #可视化dataset
    rows = []
    for i in range(3):
        example = dataset[i]
        user_msg = next(m['content'] for m in example['messages'] if m['role'] == 'user')
        assistant_msg = next(m['content'] for m in example['messages'] if m['role'] == 'assistant')
        rows.append({
            'User Message': user_msg,
            'Assistant Message': assistant_msg,
        })
    
    #以表格形式显示
    df = pd.DataFrame(rows)
    pd.set_option('display.max_colwidth', None)
    print(df)

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

del model, tokenizer
