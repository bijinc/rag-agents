"""
LangGraph node: Faithfulness Gate

Uses a separate LLM to check whether the generated answer is faithful to the retrieved context.
Answers that fail the check are flagged with low confidence — they are still returned so the caller can decide what to do.
"""

from openai import OpenAI
from src.retrieval import RetrievedChunk
from src.agentic.nodes.llm_utils import llm_json_call, append_node_error_diagnostics
from src.agentic.nodes.node_schemas import FaithfulnessResponse
from src.constants import EVALUATOR_MODEL

##############################################################################
#                              PROMPTS                                       #
##############################################################################

_SYSTEM = """
You are a financial fact-checker. Given a question, retrieved source passages, and a generated answer, determine whether the answer is faithful to the passages.

Use a practical standard, not an overly strict one.

Definition of faithful:
- The answer is faithful if its material claims are supported by the passages, even if wording is paraphrased.
- Minor stylistic rephrasing, harmless compression, and equivalent terminology are allowed.
- The answer is NOT faithful if it introduces unsupported material facts (especially specific numbers, dates, entity actions, causal claims, or comparisons).

Judgment policy:
- Prioritize factual support over exact wording.
- If a claim is partially supported but missing key specifics, treat those specifics as unsupported.
- If evidence is ambiguous or incomplete, set confidence to "low".

Output a JSON object:
- faithful: true | false
- confidence: "high" (clear judgment) | "low" (ambiguous)
- reasoning: one sentence; if false, name the key unsupported claim; if true, write "All material claims supported by passages"

Output ONLY valid JSON. No markdown fences, no prose.
"""

_USER = """
QUESTION: {question}

RETRIEVED PASSAGES:
{context}

GENERATED ANSWER:
{answer}

Is this answer faithful to the passages?
"""


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


def faithfulness_gate_node(state: dict, client: OpenAI, model: str = EVALUATOR_MODEL) -> dict:
    """
    Check whether the generated answer is grounded in the retrieved context.

    Returns updates for: is_faithful, faithfulness_status, confidence, faithfulness_reasoning
    """
    question = state["question"]
    answer   = state.get("answer", "")
    chunks   = state.get("retrieved_chunks", [])

    context_preview = _format_context_preview(chunks)

    try:
        verdict = llm_json_call(
            client,
            model=model,
            max_tokens=200,
            schema=FaithfulnessResponse,
            required_keys={"faithful", "confidence", "reasoning"},
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
        is_faithful = bool(verdict.faithful)
        confidence  = verdict.confidence
        reasoning   = verdict.reasoning

        status = "FAITHFUL" if is_faithful else "NOT FAITHFUL"
        print(f"  [faithfulness_gate] {status} (confidence={confidence}) — {reasoning}")

    except Exception as e:
        print(f"  [faithfulness_gate] WARNING: LLM failed ({e}), marking unknown")
        is_faithful = None
        faithfulness_status = "unknown"
        confidence  = "low"
        reasoning   = f"Gate unavailable: {e}"
        node_errors = dict(state.get("node_errors", {}))
        node_errors["faithfulness_gate"] = str(e)
        diagnostics = append_node_error_diagnostics(state, "faithfulness_gate", e)
        return {
            "is_faithful":            is_faithful,
            "faithfulness_status":    faithfulness_status,
            "confidence":             confidence,
            "faithfulness_reasoning": reasoning,
            "node_errors":            node_errors,
            "diagnostics":            diagnostics,
        }

    faithfulness_status = "passed" if is_faithful else "failed"
    return {
        "is_faithful":              is_faithful,
        "faithfulness_status":      faithfulness_status,
        "confidence":               confidence,
        "faithfulness_reasoning":   reasoning,
    }
