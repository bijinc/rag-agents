
import time
from openai import OpenAI

def llm_call(client: OpenAI, *, model: str, messages: list, max_tokens: int, temperature: float = 0.0) -> str:
    """
    Call the LLM and return the response text.
    Retries once (after a 2-second sleep) if the response is empty.
    """
    for attempt in range(2):
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )
        content = response.choices[0].message.content or ""
        content = content.strip()
        if content:
            return content
        if attempt == 0:
            time.sleep(2)

    return ""
