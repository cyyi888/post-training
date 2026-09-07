from __future__ import annotations

import torch


def generate_response(
    model,
    tokenizer,
    user_message: str,
    system_message: str | None = None,
    max_new_tokens: int = 64,
) -> str:
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
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
