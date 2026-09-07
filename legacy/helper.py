import numpy as np
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer


def same_seeds(seed):  # 固定随机种子（CPU）
    torch.manual_seed(seed)  # 固定随机种子（GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 为当前GPU设置
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    torch.backends.cudnn.benchmark = False  # GPU、网络结构固定，可设置为True
    torch.backends.cudnn.deterministic = True  # 固定网络结构


def generate_response(model, tokenizer, user_message, system_message=None, max_new_tokens=64):
    # 用 chat template 把对话格式化成模型期望的 prompt
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})

    # 单轮：用户一问，模型一答
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
            "Assistant: {% generation %}{{ message['content'] }}{{ eos_token }}{% endgeneration %}\n"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}Assistant: {% endif %}"
        )

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def display_dataset(dataset, n=3, max_chars=80):
    """预览前 n 条对话样本（截断长文本，避免终端折行乱成一团）。"""
    def _shorten(text):
        text = " ".join(str(text).split())  # 去掉换行，方便表格对齐
        return text if len(text) <= max_chars else text[: max_chars - 3] + "..."

    rows = []
    for i in range(min(n, len(dataset))):
        example = dataset[i]
        user_msg = next(m["content"] for m in example["messages"] if m["role"] == "user")
        assistant_msg = next(
            m["content"] for m in example["messages"] if m["role"] == "assistant"
        )
        rows.append(
            {
                "User Message": _shorten(user_msg),
                "Assistant Message": _shorten(assistant_msg),
            }
        )

    df = pd.DataFrame(rows)
    print("\n=== Dataset preview ===")
    print(df.to_string(index=True))
