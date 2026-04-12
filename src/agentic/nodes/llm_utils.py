
import time
from typing import Any
from openai import OpenAI

def llm_call(
    client: OpenAI,
    *,
    model: str,
    messages: list,
    max_tokens: int,
    temperature: float = 0.0,
    timeout_sec: float = 30.0,
    max_retries: int = 3,
    base_backoff_sec: float = 1.0,
) -> str:
    """
    Call the LLM and return response text.
    Retries with exponential backoff on failures and empty responses.
    """
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
                timeout=timeout_sec,
            )
            content = (response.choices[0].message.content or "").strip()
            if content:
                return content
            raise RuntimeError("Empty response from model")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == max_retries - 1:
                break
            time.sleep(base_backoff_sec * (2 ** attempt))

    raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_error}")


def parse_json_response(raw: str, required_keys: set[str] | None = None) -> dict[str, Any]:
    """Parse JSON response and validate required keys."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text[3:].lstrip()
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
        if text.endswith("```"):
            text = text[:-3].rstrip()

    import json

    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object")

    if required_keys:
        missing = sorted(k for k in required_keys if k not in data)
        if missing:
            raise ValueError(f"Missing required keys: {', '.join(missing)}")

    return data
