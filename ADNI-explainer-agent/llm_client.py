from ollama import chat

GPT_MODEL = "gpt-oss:20b"
QWEN_MODEL = "qwen3:30b"

MODEL_CONFIGS = {
    GPT_MODEL: {
        "think": "medium",
        "num_ctx": 49152,
        "num_predict": 24576,
        "debug_label": "GPT-OSS",
    },
    QWEN_MODEL: {
        "think": True,
        "num_ctx": 65536,
        "num_predict": 16384,
        "debug_label": "QWEN",
    },
}


# Sampling options shared by every model.
TEMPERATURE = 0
SEED = 42

# Print the answer token by token as the model generates it.
STREAM = True

# Also print the model's thinking while streaming (it can be long).
SHOW_THINKING = True

# Details of the most recent call (done_reason, lengths), for run logs.
LAST_CALL_INFO = {}


# Everything that controls generation for one model, as sent to
# Ollama. Saved with each explanation run; holds no credentials.


def get_generation_settings(model: str, overrides: dict = None) -> dict:
    config = {**MODEL_CONFIGS[model], **(overrides or {})}

    return {
        "backend": "ollama (local)",
        "model": model,
        "think": config["think"],
        "options": {
            "temperature": TEMPERATURE,
            "seed": SEED,
            "num_ctx": config["num_ctx"],
            "num_predict": config["num_predict"],
        },
    }


def ask_model(prompt: str, model: str, overrides: dict = None) -> str:
    config = MODEL_CONFIGS[model]
    settings = get_generation_settings(model, overrides)

    stream = chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        think=settings["think"],
        stream=True,
        options=settings["options"],
    )

    thinking_parts = []
    content_parts = []
    done_reason = None
    section = None

    for chunk in stream:

        thinking_piece = chunk.message.thinking or ""
        content_piece = chunk.message.content or ""

        thinking_parts.append(thinking_piece)
        content_parts.append(content_piece)

        if STREAM and thinking_piece and SHOW_THINKING:
            if section != "thinking":
                print("\n[THINKING]", flush=True)
                section = "thinking"
            print(thinking_piece, end="", flush=True)

        if STREAM and content_piece:
            if section != "answer":
                print("\n\n[ANSWER]", flush=True)
                section = "answer"
            print(content_piece, end="", flush=True)

        if chunk.done:
            done_reason = chunk.done_reason

    thinking = "".join(thinking_parts)
    content = "".join(content_parts)

    print(f"\n\n--- {config['debug_label']} DEBUG ---")
    print("Thinking length:", len(thinking))
    print("Final answer length:", len(content))
    print("Done reason:", done_reason)
    print("---------------------\n")

    LAST_CALL_INFO.clear()
    LAST_CALL_INFO.update(
        {
            "model": model,
            "done_reason": done_reason,
            "thinking_characters": len(thinking),
            "answer_characters": len(content),
        }
    )

    return content


def ask_gpt(prompt: str) -> str:
    return ask_model(prompt, GPT_MODEL)


def ask_qwen(prompt: str) -> str:
    return ask_model(prompt, QWEN_MODEL)


# ask(prompt) for the Explanation Agent. It uses the same MODEL_CONFIGS
# as the Decision Agent; there are no explanation-specific limits.


def get_explanation_ask(model: str):
    return lambda prompt: ask_model(prompt, model)


ASK_FUNCTIONS = {
    GPT_MODEL: ask_gpt,
    QWEN_MODEL: ask_qwen,
}

if __name__ == "__main__":
    answer = ask_gpt("Explain ROC AUC in one sentence")
    print(answer)
