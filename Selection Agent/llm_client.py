from ollama import chat

def ask_model(
    prompt,
    model_name,
    think,
    num_ctx,
    num_predict,
    temperature,
    seed,
):
    response = chat(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        think=think,
        stream=False,
        options={
            "temperature": temperature,
            "seed": seed,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        }
    )

    content = response.message.content

    if not content or not content.strip():
        raise RuntimeError(f"Empty response from {model_name}")

    return content
