import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **model_input,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_len = model_input["input_ids"].shape[1]
    generated_ids = outputs[0][input_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return response


def test_model_with_questions(model, tokenizer, questions, system_message=None, title="Model Output"):
    print(f"\n==={title}===")
    for i, question in enumerate(questions, 1):
        response = generate_response(model, tokenizer, question, system_message)
        print(f"\nModel Input {i}:\n{question}\nModel Output {i}:\n{response}")


def load_model_and_tokenizer(model_name, use_gpu=True):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if use_gpu and torch.cuda.is_available():
        model.to("cuda")
    elif use_gpu:
        print("Warning: CUDA unavailable, using CPU.")

    # eos 必须放进 generation 块，SFT 才会学到“答完就停”
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
