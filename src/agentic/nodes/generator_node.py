"""
LangGraph node: Generator

Formats retrieved chunks into a numbered context block and calls the LLM to produce an answer.
Uses the same grounding prompt as the baseline RAGPipeline to ensure a fair comparison.
"""

from openai import OpenAI
import re
from src.retrieval import RetrievedChunk
from src.agentic.nodes.llm_utils import llm_call
from src.agentic.nodes.llm_utils import append_node_error_diagnostics


SYSTEM_PROMPT = """
You are a financial analyst assistant. Answer using ONLY the provided source passages.

Follow these rules in order:
1) Grounding only
- Use only facts explicitly present in the passages.
- Do not use prior knowledge, assumptions, or outside information.

2) Sufficiency check
- If the passages are missing required facts, respond with exactly:
    "The retrieved context does not contain sufficient information to answer this question."
- Do not add extra text when giving this insufficiency response.

3) Citations and traceability
- Every factual claim must be attributable to at least one passage.
- Include a source-style citation in prose (for example: "According to AAPL 10-K (2024-11-01)...").
- If multiple sources support different parts, cite each relevant source.

4) Numerical fidelity
- For numbers (currency, percentages, growth rates, dates), copy values exactly as written.
- Do not round, normalize units, or reconcile conflicting numbers unless the passages explicitly do so.

5) Conflicts and ambiguity
- If passages conflict or are ambiguous, state that clearly and cite both sides.
- Never invent a reconciliation.

6) Output style
- Be concise, direct, and finance-specific.
- Prefer a short answer first, then brief supporting evidence with citations.
"""

MAX_TOKENS = 1024


def _format_context(chunks: list[RetrievedChunk]) -> str:
    """Numbered context block — same format as baseline pipeline."""
    if not chunks:
        return "(no context retrieved)"
    parts = []
    for i, c in enumerate(chunks, 1):
        if c.source_type == "sec_filing":
            label = f"{c.ticker} {c.filing_type} ({c.filing_date})"
        else:
            label = f"{c.ticker} Earnings Call ({c.period})"
        parts.append(f"[{i}] {label}\n{c.text.strip()}")
    return "\n\n".join(parts)


def _extract_numbers(text: str) -> set[str]:
    """Extract common financial number shapes for rough contradiction checks."""
    pattern = r"\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?\s?(?:billion|million|bn|m|%)?"
    return {m.strip().lower() for m in re.findall(pattern, text, flags=re.IGNORECASE) if m.strip()}


def _detect_conflicts(chunks: list[RetrievedChunk]) -> list[dict]:
    """Flag likely conflicting numeric claims for the same ticker/source period."""
    grouped: dict[tuple[str, str], list[RetrievedChunk]] = {}
    for c in chunks:
        period_key = c.filing_date or c.period or "unknown-period"
        grouped.setdefault((c.ticker, period_key), []).append(c)

    conflicts: list[dict] = []
    for (ticker, period_key), group in grouped.items():
        if len(group) < 2:
            continue
        numeric_sets = []
        for c in group:
            nums = _extract_numbers(c.text[:1200])
            if nums:
                numeric_sets.append(nums)
        if len(numeric_sets) >= 2:
            union_count = len(set().union(*numeric_sets))
            intersection_count = len(set.intersection(*numeric_sets))
            if union_count > 0 and intersection_count == 0:
                conflicts.append({
                    "ticker": ticker,
                    "period": period_key,
                    "chunk_count": len(group),
                })
    return conflicts


def _build_answer_citations(chunks: list[RetrievedChunk], max_items: int = 8) -> list[dict]:
    citations: list[dict] = []
    for c in sorted(chunks, key=lambda x: x.score, reverse=True)[:max_items]:
        citations.append(
            {
                "chunk_id": c.chunk_id,
                "score": round(float(c.score), 6),
                "ticker": c.ticker,
                "source_type": c.source_type,
                "filing_type": c.filing_type,
                "filing_date": c.filing_date,
                "period": c.period,
                "section_type": c.section_type,
                "chunk_kind": c.chunk_kind,
                "speaker_count": c.speaker_count,
                "canonical_period": c.canonical_period,
                "doc_id": c.doc_id,
            }
        )
    return citations


def _question_metric_terms(question: str) -> list[str]:
    q = question.lower()
    terms: list[str] = []
    catalog = [
        "total net revenue",
        "total revenue",
        "revenue",
        "net sales",
        "gross margin percentage",
        "gross margin",
        "operating margin",
        "operating income",
        "net income",
        "eps",
    ]
    for item in catalog:
        if item in q:
            terms.append(item)

    expanded = set(terms)
    if "revenue" in expanded or "total revenue" in expanded or "total net revenue" in expanded:
        expanded.update({"net sales", "total net sales", "revenue"})
    if "gross margin" in expanded or "gross margin percentage" in expanded:
        expanded.update({"gross margin percentage", "gross margin"})

    return sorted(expanded)


def _is_direct_lookup_question(question: str) -> bool:
    q = question.lower()
    compare_markers = ["compare", "change", "versus", "vs", "between", "trend", "from", "to"]
    management_markers = ["management", "commentary", "earnings call", "prepared remarks", "q&a"]
    if any(re.search(rf"\b{re.escape(m)}\b", q) for m in compare_markers):
        return False
    if any(re.search(rf"\b{re.escape(m)}\b", q) for m in management_markers):
        return False
    return True


def _question_period_constraints(question: str) -> tuple[str | None, str | None]:
    q = question.lower()
    m_q = re.search(r"\bq([1-4])\b", q)
    m_y = re.search(r"\b(?:fy\s*)?(20\d{2})\b", q)
    qtr = f"Q{m_q.group(1)}" if m_q else None
    year = m_y.group(1) if m_y else None
    return year, qtr


def _extract_metric_answer(question: str, chunks: list[RetrievedChunk]) -> tuple[str | None, dict | None]:
    if not _is_direct_lookup_question(question):
        return None, None

    terms = _question_metric_terms(question)
    if not terms:
        return None, None

    question_year, question_quarter = _question_period_constraints(question)
    revenue_like = any(t in terms for t in ["revenue", "total revenue", "total net revenue", "net sales", "total net sales"])
    margin_like = any("margin" in t for t in terms)

    # Metric-specific patterns reduce accidental captures like section numbers.
    percent_re = re.compile(r"\d{1,3}(?:\.\d+)?\s?%")
    money_re = re.compile(
        r"(?:\$\s?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?:\s?(?:million|billion|bn|m))?)"
        r"|(?:\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\s?(?:million|billion|bn|m)\b)",
        re.IGNORECASE,
    )
    candidates: list[tuple[float, str, RetrievedChunk]] = []

    for c in chunks[:20]:
        text = c.text
        lower = text.lower()
        if not any(t in lower for t in terms):
            continue
        if c.section_type not in {"financial_statements", "general"}:
            continue

        year_matches = True
        quarter_matches = True
        if question_year and c.fiscal_year and str(c.fiscal_year) != question_year:
            year_matches = False
        if question_quarter and c.fiscal_quarter and str(c.fiscal_quarter).upper() != question_quarter:
            quarter_matches = False

        if not year_matches or not quarter_matches:
            continue

        numeric_pattern = percent_re if margin_like else money_re if revenue_like else None
        if numeric_pattern is None:
            continue

        # Only keep values close to metric terms to avoid random table indices.
        for term in terms:
            start = 0
            while True:
                pos = lower.find(term, start)
                if pos == -1:
                    break
                win_left = max(0, pos - 60)
                win_right = min(len(text), pos + len(term) + 120)
                window = text[win_left:win_right]
                for m in numeric_pattern.finditer(window):
                    val = m.group(0).strip()
                    if not val:
                        continue

                    score = float(c.score)
                    if margin_like:
                        score += 0.2
                    if revenue_like:
                        score += 0.2
                    if c.chunk_kind == "table":
                        score += 0.1
                    if question_year and c.fiscal_year and str(c.fiscal_year) == question_year:
                        score += 0.05
                    if question_quarter and c.fiscal_quarter and str(c.fiscal_quarter).upper() == question_quarter:
                        score += 0.05
                    if term in {"total net revenue", "total revenue", "gross margin percentage"}:
                        score += 0.1

                    candidates.append((score, val, c))
                start = pos + len(term)

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, best_value, best_chunk = candidates[0]

    src = f"{best_chunk.ticker} {best_chunk.filing_type} ({best_chunk.filing_date})" if best_chunk.source_type == "sec_filing" else f"{best_chunk.ticker} Earnings Call ({best_chunk.period})"
    answer = f"{best_value}. According to {src}."
    evidence = {
        "value": best_value,
        "chunk_id": best_chunk.chunk_id,
        "source": src,
        "section_type": best_chunk.section_type,
        "chunk_kind": best_chunk.chunk_kind,
    }
    return answer, evidence


def generator_node(state: dict, client: OpenAI, model: str) -> dict:
    """
    Generate an answer from retrieved chunks using the grounded system prompt.

    Returns updates for: answer
    """
    question = state["question"]
    chunks   = state.get("retrieved_chunks", [])
    diagnostics = dict(state.get("diagnostics", {}))
    sufficiency_checks = diagnostics.get("sufficiency_checks", []) if isinstance(diagnostics, dict) else []
    repeated_insufficient = (
        len(sufficiency_checks) >= 2
        and all(check.get("decision") == "insufficient" for check in sufficiency_checks)
    )

    conflicts = _detect_conflicts(chunks)
    if conflicts:
        print(f"  [generator] WARNING: detected {len(conflicts)} potential context conflict group(s)")
    diagnostics["generator_conflicts"] = conflicts
    diagnostics["answer_citations"] = _build_answer_citations(chunks)

    extracted_answer, extracted_evidence = _extract_metric_answer(question, chunks)
    if extracted_answer:
        diagnostics["generator_extraction"] = {
            "used": True,
            "evidence": extracted_evidence,
        }
        print("  [generator] Used deterministic metric extraction path")
        return {"answer": extracted_answer, "diagnostics": diagnostics}
    diagnostics["generator_extraction"] = {"used": False}

    # if repeated_insufficient and conflicts:
    #     print("  [generator] Conservative fallback: repeated insufficiency + conflicts -> refusal")
    #     return {
    #         "answer": "The retrieved context does not contain sufficient information to answer this question.",
    #         "diagnostics": diagnostics,
    #     }

    context_str  = _format_context(chunks)
    conflict_instruction = ""
    if conflicts:
        conflict_instruction = (
            "\n\nAdditional instruction: Retrieved passages may contain conflicting figures for similar periods. "
            "When this happens, explicitly note uncertainty and avoid choosing unsupported figures."
        )
    user_message = f"Source passages:\n\n{context_str}\n\nQuestion: {question}{conflict_instruction}"

    try:
        answer = llm_call(
            client,
            model=model,
            max_tokens=MAX_TOKENS,
            temperature=0.0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            timeout_sec=45.0,
            max_retries=3,
        )
    except Exception as exc:  # noqa: BLE001
        answer = "The retrieved context does not contain sufficient information to answer this question."
        node_errors = dict(state.get("node_errors", {}))
        node_errors["generator"] = str(exc)
        diagnostics = append_node_error_diagnostics(state, "generator", exc)
        print(f"  [generator] WARNING: generation failed ({exc})")
        return {"answer": answer, "node_errors": node_errors, "diagnostics": diagnostics}

    print(f"  [generator] Generated {len(answer)} chars using {len(chunks)} chunks")

    return {"answer": answer, "diagnostics": diagnostics}
