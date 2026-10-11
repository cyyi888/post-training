from __future__ import annotations

import torch

from src.data.chat_templates import render_chat


def generate_response(
    model,
    tokenizer,
    user_message: str,
    system_message: str | None = None,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    top_p: float = 1.0,
    do_sample: bool | None = None,
) -> str:
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_message})

    prompt = render_chat(tokenizer, messages, add_generation_prompt=True)

    model_input = tokenizer(prompt, return_tensors="pt").to(model.device)
    sample = bool(do_sample) if do_sample is not None else float(temperature) > 0.0
    gen_kwargs: dict = {
        "max_new_tokens": max_new_tokens,
        "do_sample": sample,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if sample:
        gen_kwargs["temperature"] = max(float(temperature), 1e-5)
        gen_kwargs["top_p"] = float(top_p)

    with torch.no_grad():
        outputs = model.generate(**model_input, **gen_kwargs)

    input_len = model_input["input_ids"].shape[1]
    generated_ids = outputs[0][input_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def test_model_with_questions(
    model,
    tokenizer,
    questions: list[str],
    system_message: str | None = None,
    title: str = "Model Output",
) -> None:
    print(f"\n==={title}===")
    for i, question in enumerate(questions, 1):
        response = generate_response(model, tokenizer, question, system_message)
        print(f"\nModel Input {i}:\n{question}\nModel Output {i}:\n{response}")
