"""src/nodes/sufficiency_checker.py

LangGraph node: Sufficiency Checker

Asks the LLM whether the retrieved chunks contain enough information
to answer the question. Also provides the conditional edge routing
function used by the LangGraph graph.
"""

import json
from openai import OpenAI
from src.retrieval import RetrievedChunk
from src.agentic.nodes.llm_utils import llm_call

##############################################################################
#                              PROMPTS                                       #
##############################################################################

_SYSTEM = """You are evaluating whether a set of retrieved financial document passages contains enough information to answer a question.

Be pragmatic: if the passages contain the key facts needed (even if not perfectly phrased), mark as sufficient.
Mark as insufficient only if critical information is clearly absent (e.g., a specific quarter's figure is asked but no relevant passage mentions it).

Output a JSON object:
- sufficient: true | false
- reason: one sentence explaining your judgment

Output ONLY valid JSON. No markdown fences, no prose."""

_USER = """QUESTION: {question}

RETRIEVED PASSAGES (preview):
{context}

Do these passages contain enough information to answer the question?"""


##############################################################################
#                              HELPERS                                       #
##############################################################################

def _format_context_preview(chunks: list[RetrievedChunk], max_chars: int = 3000) -> str:
    """Abbreviated context for the sufficiency prompt — avoids huge token usage."""
    parts = []
    total = 0
    for i, c in enumerate(chunks, 1):
        label = (f"{c.ticker} {c.filing_type} ({c.filing_date})"
                 if c.source_type == "sec_filing"
                 else f"{c.ticker} ECT ({c.period})")
        preview = c.text[:600].replace("\n", " ").strip()
        entry = f"[{i}] {label}: {preview}..."
        parts.append(entry)
        total += len(entry)
        if total >= max_chars:
            break
    return "\n".join(parts)


##############################################################################
#                              NODE                                          #
##############################################################################

def sufficiency_checker_node(state: dict, client: OpenAI) -> dict:
    """
    Evaluate whether retrieved chunks are sufficient to answer the question.

    Returns updates for: sufficiency ("sufficient" | "insufficient")
    """
    question = state["question"]
    chunks   = state.get("retrieved_chunks", [])

    if not chunks:
        print("  [sufficiency_checker] No chunks retrieved → insufficient")
        return {"sufficiency": "insufficient"}

    context_preview = _format_context_preview(chunks)

    try:
        raw = llm_call(
            client,
            model="qwen/qwen-2.5-7b-instruct",
            max_tokens=150,
            temperature=0.0,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": _USER.format(
                    question=question,
                    context=context_preview,
                )},
            ],
        )
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.rstrip("`").strip()

        verdict      = json.loads(raw)
        is_sufficient = bool(verdict.get("sufficient", True))
        reason        = verdict.get("reason", "")

        result = "sufficient" if is_sufficient else "insufficient"
        print(f"  [sufficiency_checker] {result} — {reason}")
        return {"sufficiency": result}

    except Exception as e:
        # On LLM failure, default to sufficient to avoid infinite loops
        print(f"  [sufficiency_checker] WARNING: LLM failed ({e}), defaulting to sufficient")
        return {"sufficiency": "sufficient"}


##############################################################################
#                         CONDITIONAL EDGE ROUTING                          #
##############################################################################

def route_sufficiency(state: dict) -> str:
    """
    Conditional edge function for LangGraph.

    Routes to "query_refiner" if insufficient AND retries remain,
    otherwise to "generator".
    """
    sufficiency        = state.get("sufficiency", "sufficient")
    retrieval_attempts = state.get("retrieval_attempts", 0)

    if sufficiency == "insufficient" and retrieval_attempts < 2:
        return "insufficient"
    return "sufficient"
