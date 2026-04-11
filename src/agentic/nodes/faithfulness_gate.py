"""
LangGraph node: Faithfulness Gate

Uses a separate LLM to check whether the generated answer is faithful to the retrieved context.
Answers that fail the check are flagged with low confidence — they are still returned so the caller can decide what to do.
"""

import json
from openai import OpenAI
from src.retrieval import RetrievedChunk
from src.agentic.nodes.llm_utils import llm_call
from src.constants import EVALUATOR_MODEL

##############################################################################
#                              PROMPTS                                       #
##############################################################################

_SYSTEM = """You are a financial fact-checker. Given a question, retrieved source passages, and a generated answer, determine whether the answer is faithful to the passages.

Definition of faithful: every factual claim in the answer can be directly traced to one of the provided passages. The answer does not introduce numbers, dates, or statements absent from the passages.

Output a JSON object:
- faithful: true | false
- confidence: "high" (clear judgment) | "low" (ambiguous)
- reasoning: one sentence — name any specific unsupported claim, or write "All claims supported by passages"

Output ONLY valid JSON. No markdown fences, no prose."""

_USER = """QUESTION: {question}

RETRIEVED PASSAGES:
{context}

GENERATED ANSWER:
{answer}

Is this answer faithful to the passages?"""

##############################################################################
#                              HELPERS                                       #
##############################################################################

def _format_context_preview(chunks: list[RetrievedChunk], max_chars: int = 4000) -> str:
    """Context preview for the faithfulness prompt."""
    parts = []
    total = 0
    for i, c in enumerate(chunks, 1):
        label = (f"{c.ticker} {c.filing_type} ({c.filing_date})"
                 if c.source_type == "sec_filing"
                 else f"{c.ticker} ECT ({c.period})")
        text_preview = c.text[:400].strip()
        entry = f"[{i}] {label}:\n{text_preview}"
        parts.append(entry)
        total += len(entry)
        if total >= max_chars:
            break
    return "\n\n".join(parts)


##############################################################################
#                              NODE                                          #
##############################################################################

def faithfulness_gate_node(state: dict, client: OpenAI) -> dict:
    """
    Check whether the generated answer is grounded in the retrieved context.

    Returns updates for: is_faithful, confidence, faithfulness_reasoning
    """
    question = state["question"]
    answer   = state.get("answer", "")
    chunks   = state.get("retrieved_chunks", [])

    context_preview = _format_context_preview(chunks)

    try:
        raw = llm_call(
            client,
            model=EVALUATOR_MODEL,
            max_tokens=200,
            temperature=0.0,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": _USER.format(
                    question=question,
                    context=context_preview,
                    answer=answer,
                )},
            ],
        )
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.rstrip("`").strip()

        verdict     = json.loads(raw)
        is_faithful = bool(verdict.get("faithful", True))
        confidence  = verdict.get("confidence", "high")
        reasoning   = verdict.get("reasoning", "")

        status = "FAITHFUL" if is_faithful else "NOT FAITHFUL"
        print(f"  [faithfulness_gate] {status} (confidence={confidence}) — {reasoning}")

    except Exception as e:
        # Gate failure: assume faithful with low confidence rather than blocking
        print(f"  [faithfulness_gate] WARNING: LLM failed ({e}), defaulting to faithful/low")
        is_faithful = True
        confidence  = "low"
        reasoning   = f"Gate unavailable: {e}"

    return {
        "is_faithful":           is_faithful,
        "confidence":            confidence,
        "faithfulness_reasoning": reasoning,
    }
