"""
LangGraph node: Sufficiency Checker

Asks the LLM whether the retrieved chunks contain enough information to answer the question.
Also provides the conditional edge routing function used by the LangGraph graph.
"""

import re

from openai import OpenAI
from src.retrieval import RetrievedChunk
from src.agentic.nodes.llm_utils import llm_json_call, append_node_error_diagnostics
from src.agentic.nodes.node_schemas import SufficiencyResponse

##############################################################################
#                              PROMPTS                                       #
##############################################################################

_SYSTEM = """
You are a context-sufficiency judge for a financial RAG system.

Task:
Decide whether the retrieved passages are sufficient to answer the user question using only grounded evidence from the passages.

Decision policy:
- Return sufficient=true only when the passages contain enough specific evidence to answer the question directly.
- Return sufficient=false if any critical piece is missing, ambiguous, contradictory, or only implied.
- Do not assume facts not explicitly present.
- Do not use outside knowledge.
- If the question asks for a specific company, period, metric, filing type, or management commentary, treat those as required constraints.
- For comparison/trend questions, evidence must cover all compared items/periods.
- For management-commentary questions, prefer explicit commentary evidence (not just numeric filings unless clearly sufficient).
- If passages are mostly generic or off-topic, mark insufficient.
- For table-heavy SEC passages, treat evidence as sufficient when the requested metric and period are present, even if formatting is noisy.

What counts as sufficient:
- Exact or near-exact metric/value and matching entity/timeframe.
- Clear textual evidence that directly addresses the question intent.
- Enough coverage to avoid guesswork.

What counts as insufficient:
- Missing timeframe/ticker/source alignment.
- Partial coverage (only one side of a compare question).
- Evidence too weak to support a confident grounded answer.
- Conflicting passages without a clear resolution.

Output format (JSON only):
{
  "sufficient": true | false,
  "reason": "One concise sentence citing the key sufficiency or missing element."
}

Output rules:
- Output ONLY valid JSON.
- No markdown, no code fences, no extra keys, no extra text.
"""

_USER = """
QUESTION: {question}

RETRIEVED PASSAGES (preview):
{context}

Do these passages contain enough information to answer the question?
"""


##############################################################################
#                              HELPERS                                       #
##############################################################################

def _format_context_preview(chunks: list[RetrievedChunk], max_chars: int = 8000) -> str:
    """Abbreviated context for the sufficiency prompt — avoids huge token usage."""
    parts = []
    total = 0
    for i, c in enumerate(chunks, 1):
        label = (f"{c.ticker} {c.filing_type} ({c.filing_date})"
                 if c.source_type == "sec_filing"
                 else f"{c.ticker} ECT ({c.period})")
        # Use longer preview for table chunks since financial numbers are often
        # spread across wide fixed-width rows that get cut off at 600 chars.
        preview_len = 1000 if c.chunk_kind == "table" else 600
        preview = c.text[:preview_len].replace("\n", " ").strip()
        entry = f"[{i}] {label}: {preview}..."
        parts.append(entry)
        total += len(entry)
        if total >= max_chars:
            break
    return "\n".join(parts)


def _compute_coverage(chunks: list[RetrievedChunk], sub_questions: list[str]) -> dict:
    """Lightweight retrieval coverage signals used before the LLM sufficiency call."""
    tickers = {c.ticker for c in chunks if c.ticker}
    source_types = {c.source_type for c in chunks if c.source_type}
    filing_dates = {c.filing_date for c in chunks if c.filing_date}
    periods = {c.period for c in chunks if c.period}
    sec_chunks = sum(1 for c in chunks if c.source_type == "sec_filing")
    ect_chunks = sum(1 for c in chunks if c.source_type == "ect")
    return {
        "chunk_count": len(chunks),
        "sub_question_count": len(sub_questions),
        "ticker_count": len(tickers),
        "source_type_count": len(source_types),
        "filing_date_count": len(filing_dates),
        "period_count": len(periods),
        "sec_chunk_count": sec_chunks,
        "ect_chunk_count": ect_chunks,
    }


def _looks_underspecified(question: str, coverage: dict) -> bool:
    """Fast pre-check for obvious insufficiency without spending another LLM call."""
    q = question.lower()
    asks_compare = any(token in q for token in ["compare", "change", "vs", "versus", "between"])
    asks_management = any(token in q for token in ["management", "said", "commentary", "earnings call"])
    asks_time = any(token in q for token in ["q1", "q2", "q3", "q4", "fy", "202", "201"])

    if coverage["chunk_count"] < 2:
        return True
    if asks_compare and coverage["chunk_count"] < 3:
        return True
    if asks_management and coverage["ect_chunk_count"] == 0:
        return True
    if asks_time and (coverage["filing_date_count"] + coverage["period_count"]) == 0:
        return True
    return False


def _question_flags(question: str) -> dict[str, bool]:
    q = question.lower()

    def has_any(patterns: list[str]) -> bool:
        return any(re.search(p, q) for p in patterns)

    return {
        "asks_compare": has_any([
            r"\bcompare\b",
            r"\bchange\b",
            r"\bvs\b",
            r"\bversus\b",
            r"\bbetween\b",
            r"\btrend\b",
            r"\bfrom\b",
            r"\bto\b",
        ]),
        "asks_management": has_any([
            r"\bmanagement\b",
            r"\bsaid\b",
            r"\bcommentary\b",
            r"\bearnings\s+call\b",
            r"\bprepared\s+remarks\b",
            r"\bq\s*&\s*a\b",
        ]),
        "asks_time": has_any([
            r"\bq[1-4]\b",
            r"\bfy\b",
            r"\b20\d{2}\b",
        ]),
    }


def _metric_signals(question: str) -> list[str]:
    q = question.lower()
    signals: list[str] = []
    catalog = [
        "total revenue",
        "revenue",
        "net sales",
        "gross margin",
        "operating margin",
        "operating income",
        "net income",
        "eps",
        "earnings per share",
        "services revenue",
        "cash flow",
        "r&d",
        "capex",
    ]
    for item in catalog:
        if item in q:
            signals.append(item)

    # Expand common financial synonyms.
    expanded = set(signals)
    if "revenue" in expanded or "total revenue" in expanded:
        expanded.update({"net sales", "total net sales"})
    if "net sales" in expanded or "total net sales" in expanded:
        expanded.update({"revenue", "total revenue"})

    signals = sorted(expanded)
    return signals


def _structured_evidence_score(question: str, chunks: list[RetrievedChunk]) -> tuple[int, list[str]]:
    metric_terms = _metric_signals(question)
    numeric_re = re.compile(r"\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?\s?(?:%|million|billion|bn|m)?", re.IGNORECASE)
    matched_ids: list[str] = []
    score = 0

    for c in chunks[:20]:
        text = c.text.lower()
        has_number = bool(numeric_re.search(text))
        has_metric = bool(metric_terms) and any(t in text for t in metric_terms)
        structural_bonus = 1 if (c.chunk_kind in {"table", "speaker_turns"} or c.section_type == "financial_statements") else 0
        if has_number and has_metric:
            score += 2 + structural_bonus
            matched_ids.append(c.chunk_id)

    return score, matched_ids


def _structured_fast_path(question: str, chunks: list[RetrievedChunk], coverage: dict, retrieval_attempts: int = 0) -> tuple[bool, str, list[str]]:
    flags = _question_flags(question)

    score, matched_ids = _structured_evidence_score(question, chunks)

    # Direct lookup questions: low bar (score >= 3)
    if not flags["asks_compare"] and not flags["asks_management"]:
        if flags["asks_time"] and (coverage["filing_date_count"] + coverage["period_count"]) < 2:
            return False, "", []
        if coverage["sec_chunk_count"] >= 1 and score >= 3:
            return True, "Structured metric evidence detected in retrieved financial chunks.", matched_ids

    # Comparison/management questions: require more evidence (score >= 4),
    # but still allow fast-path if we have substantial coverage.
    if flags["asks_compare"] and score >= 4 and coverage["chunk_count"] >= 5:
        return True, "Sufficient structured evidence for comparison question.", matched_ids

    if flags["asks_management"] and coverage["ect_chunk_count"] >= 3 and coverage["chunk_count"] >= 5:
        return True, "Sufficient ECT coverage for management commentary question.", matched_ids

    # After 1+ retrieval attempts with reasonable context, let the generator try. 
    # The faithfulness gate will catch bad answers downstream. This prevents the retry loop from diluting 
    # good initial retrieval by progressively relaxing filters and losing relevant chunks.
    if retrieval_attempts >= 1 and coverage["chunk_count"] >= 5:
        return True, "Sufficient context after retrieval retry — deferring to generator.", matched_ids

    return False, "", matched_ids


##############################################################################
#                              NODE                                          #
##############################################################################

def sufficiency_checker_node(state: dict, client: OpenAI, model: str) -> dict:
    """
    Evaluate whether retrieved chunks are sufficient to answer the question.

    Returns updates for: sufficiency ("sufficient" | "insufficient")
    """
    question = state["question"]
    chunks   = state.get("retrieved_chunks", [])
    sub_questions = state.get("sub_questions", [question])
    retrieval_attempts = state.get("retrieval_attempts", 0)

    diagnostics = dict(state.get("diagnostics", {}))
    sufficiency_checks = list(diagnostics.get("sufficiency_checks", []))

    if not chunks:
        print("  [sufficiency_checker] No chunks retrieved → insufficient")
        sufficiency_checks.append({
            "attempt": retrieval_attempts,
            "decision": "insufficient",
            "reason": "No chunks retrieved",
            "coverage": _compute_coverage([], sub_questions),
        })
        diagnostics["sufficiency_checks"] = sufficiency_checks
        return {"sufficiency": "insufficient", "diagnostics": diagnostics}

    coverage = _compute_coverage(chunks, sub_questions)

    fast_path_ok, fast_path_reason, fast_path_chunks = _structured_fast_path(question, chunks, coverage, retrieval_attempts)
    if fast_path_ok:
        print(f"  [sufficiency_checker] sufficient — {fast_path_reason}")
        sufficiency_checks.append({
            "attempt": retrieval_attempts,
            "decision": "sufficient",
            "reason": fast_path_reason,
            "coverage": coverage,
            "used_llm": False,
            "structured_fast_path": True,
            "evidence_chunk_ids": fast_path_chunks,
        })
        diagnostics["sufficiency_checks"] = sufficiency_checks
        return {"sufficiency": "sufficient", "diagnostics": diagnostics}

    if _looks_underspecified(question, coverage):
        reason = f"Heuristic insufficiency from sparse coverage: {coverage}"
        print(f"  [sufficiency_checker] insufficient — {reason}")
        sufficiency_checks.append({
            "attempt": retrieval_attempts,
            "decision": "insufficient",
            "reason": reason,
            "coverage": coverage,
            "used_llm": False,
        })
        diagnostics["sufficiency_checks"] = sufficiency_checks
        return {"sufficiency": "insufficient", "diagnostics": diagnostics}

    context_preview = _format_context_preview(chunks)

    try:
        verdict = llm_json_call(
            client,
            model=model,
            max_tokens=150,
            schema=SufficiencyResponse,
            required_keys={"sufficient", "reason"},
            temperature=0.0,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": _USER.format(
                    question=question,
                    context=context_preview,
                )},
            ],
        )
        is_sufficient = bool(verdict.sufficient)
        reason        = verdict.reason

        result = "sufficient" if is_sufficient else "insufficient"
        print(f"  [sufficiency_checker] {result} — {reason}")
        sufficiency_checks.append({
            "attempt": retrieval_attempts,
            "decision": result,
            "reason": reason,
            "coverage": coverage,
            "used_llm": True,
        })
        diagnostics["sufficiency_checks"] = sufficiency_checks
        return {"sufficiency": result, "diagnostics": diagnostics}

    except Exception as e:
        print(f"  [sufficiency_checker] WARNING: LLM failed ({e}), marking insufficient")
        node_errors = dict(state.get("node_errors", {}))
        node_errors["sufficiency_checker"] = str(e)
        diagnostics = append_node_error_diagnostics(state, "sufficiency_checker", e)
        sufficiency_checks.append({
            "attempt": retrieval_attempts,
            "decision": "insufficient",
            "reason": f"LLM evaluation failure: {e}",
            "coverage": coverage,
            "used_llm": True,
        })
        diagnostics["sufficiency_checks"] = sufficiency_checks
        return {
            "sufficiency": "insufficient",
            "node_errors": node_errors,
            "diagnostics": diagnostics,
        }


##############################################################################
#                         CONDITIONAL EDGE ROUTING                           #
##############################################################################

def route_sufficiency(state: dict) -> str:
    """
    Conditional edge function for LangGraph.
    Routes to "query_refiner" if insufficient AND retries remain, otherwise to "generator".
    """
    sufficiency        = state.get("sufficiency", "sufficient")
    retrieval_attempts = state.get("retrieval_attempts", 0)
    diagnostics        = state.get("diagnostics", {})

    # Early-stop retries when new retrieval is nearly identical to the previous one.
    retrieval_history = diagnostics.get("retrieval_history", []) if isinstance(diagnostics, dict) else []
    if sufficiency == "insufficient" and retrieval_attempts >= 1 and retrieval_history:
        latest = retrieval_history[-1]
        overlap = latest.get("overlap_with_prev")
        if overlap is not None and overlap >= 0.90:
            print(f"  [sufficiency_checker] Early stop retries due to high overlap ({overlap:.2f})")
            return "sufficient"

    if sufficiency == "insufficient" and retrieval_attempts < 3:
        return "insufficient"
    return "sufficient"
