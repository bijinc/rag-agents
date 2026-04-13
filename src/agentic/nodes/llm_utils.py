
import time
from typing import Any
from openai import OpenAI
import json
from pydantic import BaseModel, ValidationError


def _extract_content(response: Any) -> str:
    content = (response.choices[0].message.content or "").strip()
    if content:
        print(f"  [llm_call] Success - response: {content[:200]}{'...' if len(content) > 200 else ''}")
        return content
    raise RuntimeError("Empty response from model")


def _is_unsupported_response_format_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "response_format" in message and any(
        token in message
        for token in ["unsupported", "unknown", "not allowed", "extra_forbidden", "invalid parameter"]
    )

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
    response_format: dict[str, Any] | None = None,
) -> str:
    """
    Call the LLM and return response text.
    Retries with exponential backoff on failures and empty responses.
    """
    last_error: Exception | None = None
    enforce_response_format = response_format
    for attempt in range(max_retries):
        request_kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
            "timeout": timeout_sec,
        }
        if enforce_response_format is not None:
            request_kwargs["response_format"] = enforce_response_format

        try:
            response = client.chat.completions.create(**request_kwargs)
            return _extract_content(response)
        except Exception as exc:
            # Some OpenAI-compatible backends do not support `response_format`.
            # Fall back to regular completion while retaining downstream schema validation.
            if enforce_response_format is not None and _is_unsupported_response_format_error(exc):
                enforce_response_format = None
                fallback_kwargs = dict(request_kwargs)
                fallback_kwargs.pop("response_format", None)
                try:
                    response = client.chat.completions.create(**fallback_kwargs)
                    return _extract_content(response)
                except Exception as fallback_exc:
                    last_error = fallback_exc
                    if attempt == max_retries - 1:
                        break
                    time.sleep(base_backoff_sec * (2 ** attempt))
                    continue

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

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Some models prepend/append prose; recover by slicing the first JSON object.
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        data = json.loads(text[start : end + 1])

    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object")

    if required_keys:
        missing = sorted(k for k in required_keys if k not in data)
        if missing:
            raise ValueError(f"Missing required keys: {', '.join(missing)}")

    return data


def llm_json_call(
    client: OpenAI,
    *,
    model: str,
    messages: list,
    max_tokens: int,
    schema: type[BaseModel],
    required_keys: set[str] | None = None,
    temperature: float = 0.0,
    timeout_sec: float = 30.0,
    max_retries: int = 3,
) -> BaseModel:
    """Call LLM, parse JSON, and validate against a Pydantic schema."""
    raw = llm_call(
        client,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=messages,
        timeout_sec=timeout_sec,
        max_retries=max_retries,
        response_format={"type": "json_object"},
    )
    parsed = parse_json_response(raw, required_keys=required_keys)
    try:
        return schema.model_validate(parsed)
    except ValidationError as exc:
        raise RuntimeError(f"Schema validation failed: {exc}") from exc


def classify_error(exc: Exception) -> str:
    """Classify node errors for consistent diagnostics and reporting."""
    message = str(exc).lower()
    if "schema validation" in message:
        return "schema_error"
    if any(token in message for token in ["json", "expecting value", "extra data", "decode"]):
        return "parse_error"
    if any(token in message for token in ["timeout", "timed out"]):
        return "timeout_error"
    if any(token in message for token in ["rate limit", "429", "too many requests"]):
        return "rate_limit_error"
    if any(token in message for token in ["api key", "authentication", "unauthorized", "401", "403"]):
        return "auth_error"
    if "empty response" in message:
        return "empty_response_error"
    return "unknown_error"


def append_node_error_diagnostics(state: dict, node_name: str, exc: Exception) -> dict[str, Any]:
    """Append structured node error events to diagnostics."""
    diagnostics = dict(state.get("diagnostics", {}))
    events = list(diagnostics.get("node_error_events", []))
    category = classify_error(exc)
    events.append({
        "node": node_name,
        "category": category,
        "message": str(exc),
    })
    diagnostics["node_error_events"] = events
    category_counts = dict(diagnostics.get("node_error_category_counts", {}))
    category_counts[category] = int(category_counts.get(category, 0)) + 1
    diagnostics["node_error_category_counts"] = category_counts
    return diagnostics
