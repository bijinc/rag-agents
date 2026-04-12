import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_llm_json(raw: str, required_keys: set[str] | None = None) -> dict[str, Any]:
    """Parse JSON from an LLM response, tolerating markdown fences."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text[3:].lstrip()
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
        if text.endswith("```"):
            text = text[:-3].rstrip()

    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object")

    if required_keys:
        missing = sorted(k for k in required_keys if k not in data)
        if missing:
            raise ValueError(f"Missing required JSON keys: {', '.join(missing)}")

    return data


def llm_call_with_retry(
    client: OpenAI,
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float = 0.0,
    timeout_sec: float = 30.0,
    max_retries: int = 3,
    base_backoff_sec: float = 1.0,
) -> str:
    """Call the LLM with bounded retries and exponential backoff."""
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


def validate_benchmark_items(benchmark: list[dict[str, Any]]) -> None:
    """Validate benchmark structure before expensive evaluation work starts."""
    if not isinstance(benchmark, list):
        raise ValueError("Benchmark JSON must be a list of question objects")

    required = {"id", "question", "difficulty", "answer"}
    valid_difficulties = {"L1", "L2", "L3", "L4"}
    seen_ids: set[str] = set()

    for idx, item in enumerate(benchmark):
        if not isinstance(item, dict):
            raise ValueError(f"Benchmark item at index {idx} is not an object")

        missing = sorted(k for k in required if k not in item)
        if missing:
            raise ValueError(
                f"Benchmark item at index {idx} missing required keys: {', '.join(missing)}"
            )

        qid = str(item["id"])
        if qid in seen_ids:
            raise ValueError(f"Duplicate benchmark id: {qid}")
        seen_ids.add(qid)

        diff = str(item["difficulty"])
        if diff not in valid_difficulties:
            raise ValueError(
                f"Benchmark item id={qid} has invalid difficulty={diff}; expected one of {sorted(valid_difficulties)}"
            )


def validate_eval_flags(skip_ragas: bool, skip_judge: bool, judge_only: bool) -> None:
    if judge_only and skip_judge:
        raise ValueError("Invalid flags: --judge-only cannot be combined with --skip-judge")
    if judge_only and skip_ragas:
        raise ValueError("Invalid flags: --judge-only already skips pipeline/ragas; remove --skip-ragas")
    if skip_ragas and skip_judge:
        raise ValueError("Invalid flags: both --skip-ragas and --skip-judge are set; nothing to evaluate")


def load_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def save_json(path: str, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)