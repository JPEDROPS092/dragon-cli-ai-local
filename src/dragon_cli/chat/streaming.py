"""Streaming text generation via llama.cpp."""

from ..config import C_CYAN, C_RESET


def generate_streaming(llm, history: list[dict]) -> str:
    print(f"{C_CYAN}Model:{C_RESET} ", end="", flush=True)
    response = ""
    stream = llm.create_chat_completion(
        messages=history, max_tokens=1024, temperature=0.7, stream=True,
    )
    for chunk in stream:
        token = chunk["choices"][0]["delta"].get("content", "")
        if token:
            print(token, end="", flush=True)
            response += token
    print()
    return response
