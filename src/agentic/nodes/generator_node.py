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
INSUFFICIENT_ANSWER = "The retrieved context does not contain sufficient information to answer this question."


COMPLEX_PARTIAL_SYSTEM_PROMPT = """
You are a financial analyst assistant. Answer using ONLY the provided source passages.

Follow these rules in order:
1) Grounding only
- Use only facts explicitly present in the passages.
- Do not use prior knowledge, assumptions, or outside information.

2) Partial-answer policy for complex questions
- If passages support part of the question, provide that supported part.
- Explicitly label missing pieces as unavailable from the provided passages.
- If no relevant facts are present at all, respond with exactly:
    "The retrieved context does not contain sufficient information to answer this question."

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


PARTIAL_REWRITE_SYSTEM_PROMPT = """
You are a financial QA answer rewriter.

Task:
- Rewrite the draft answer so every factual claim is directly supported by the provided passages.
- Remove any unsupported numeric/date/value claims.
- Keep supported claims concise and cited.
- If no supported factual content remains after removing unsupported claims, return exactly:
    "The retrieved context does not contain sufficient information to answer this question."

Output only the rewritten answer text.
"""


def _normalize_table_text(text: str) -> str:
    """Collapse SEC financial table whitespace into a more LLM-readable format.

    Raw SEC tables use fixed-width formatting like:
        Total net sales           $         391,035           2      $         383,285
    This normalizes to:
        Total net sales  $391,035  2  $383,285

    This helps smaller LLMs (Qwen-7B) extract numbers from table chunks.
    """
    import re as _re
    # Remove [Table context: ...] prefix — useful for embeddings but noisy for LLM
    text = _re.sub(r"^\[Table context:.*?\]\n?", "", text, flags=_re.DOTALL)
    # Collapse runs of whitespace (but preserve newlines for row structure)
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        # Collapse multiple spaces into double-space (preserves column feel)
        line = _re.sub(r"  +", "  ", line)
        # Close gap between $ and number: "$  391,035" → "$391,035"
        line = _re.sub(r"\$\s+(\d)", r"$\1", line)
        if line.strip():
            cleaned.append(line.strip())
    return "\n".join(cleaned)


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
        text = c.text.strip()
        # Clean up table chunks so the LLM can parse numbers
        if c.chunk_kind == "table":
            text = _normalize_table_text(text)
        parts.append(f"[{i}] {label}\n{text}")
    return "\n\n".join(parts)


def _extract_numbers(text: str) -> set[str]:
    """Extract common financial number shapes for rough contradiction checks."""
    pattern = r"\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?\s?(?:billion|million|bn|m|%)?"
    return {m.strip().lower() for m in re.findall(pattern, text, flags=re.IGNORECASE) if m.strip()}


def _normalize_number_token(token: str) -> str:
    t = token.lower().strip()
    t = re.sub(r"\s+", " ", t)
    t = t.replace(",", "")
    t = t.replace("$ ", "$")
    return t


def _canonical_numeric_token(token: str) -> str:
    t = _normalize_number_token(token)
    t = t.replace("$", "")
    t = t.replace("%", "")
    t = re.sub(r"\b(billion|million|bn|m)\b", "", t)
    t = re.sub(r"\s+", "", t)
    return t


def _is_material_numeric_claim(token: str) -> bool:
    t = token.lower().strip()
    if not t:
        return False
    if any(mark in t for mark in ["$", "%", ",", "million", "billion", "bn"]):
        return True
    if "." in t:
        return True
    return False


def _unsupported_answer_numbers(answer: str, chunks: list[RetrievedChunk], max_chunks: int = 30) -> list[str]:
    raw_answer_numbers = {_normalize_number_token(n) for n in _extract_numbers(answer)}
    answer_numbers = {n for n in raw_answer_numbers if _is_material_numeric_claim(n)}
    if not answer_numbers:
        return []

    context_numbers: set[str] = set()
    for c in chunks[:max_chunks]:
        context_numbers.update({_canonical_numeric_token(n) for n in _extract_numbers(c.text)})

    unsupported = sorted(n for n in answer_numbers if _canonical_numeric_token(n) not in context_numbers)
    return unsupported


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

    # Segment qualifiers help row-level extraction choose the right table row.
    if "iphone" in q:
        terms.append("iphone")
    if "services" in q:
        terms.append("services")
    if "mac" in q:
        terms.append("mac")
    if "ipad" in q:
        terms.append("ipad")
    if "wearables" in q:
        terms.append("wearables")
    if "products" in q:
        terms.append("products")

    expanded = set(terms)
    if "revenue" in expanded or "total revenue" in expanded or "total net revenue" in expanded:
        expanded.update({"net sales", "total net sales", "revenue"})
    if "gross margin" in expanded or "gross margin percentage" in expanded:
        expanded.update({"gross margin percentage", "gross margin"})

    return sorted(expanded)


def _question_segment_label(question: str) -> str | None:
    q = question.lower()
    if "iphone" in q:
        return "iphone"
    if "services" in q:
        return "services"
    if "mac" in q:
        return "mac"
    if "ipad" in q:
        return "ipad"
    if "wearables" in q:
        return "wearables"
    if "products" in q:
        return "products"
    return None


def _is_direct_lookup_question(question: str) -> bool:
    q = question.lower()
    compare_markers = [
        "compare",
        "change",
        "versus",
        "vs",
        "between",
        "trend",
        "from",
        "to",
        "year-over-year",
        "yoy",
        "growth",
        "increase",
        "decrease",
        "rose",
        "fell",
    ]
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


def _is_total_revenue_question(question: str) -> bool:
    q = question.lower()
    return "total net revenue" in q or "total revenue" in q


def _line_has_total_revenue_label(line_lower: str) -> bool:
    return bool(re.search(r"\btotal\s+(?:net\s+)?(?:sales|revenue)\b", line_lower))


def _line_is_valid_metric_row(
    line_lower: str,
    question: str,
    revenue_like: bool,
    margin_like: bool,
) -> bool:
    """Filter out noisy matches and keep only row-like evidence lines."""
    if not line_lower.strip():
        return False

    if revenue_like:
        # Reject percentage/portion rows that often sit near the target metric.
        if "percentage of total net sales" in line_lower:
            return False
        if "portion of total net sales" in line_lower:
            return False

        # For total-revenue questions, only accept explicitly total rows.
        if _is_total_revenue_question(question):
            if not _line_has_total_revenue_label(line_lower):
                return False
            # Guard against segment rows that can be numerically larger in context windows.
            if re.search(r"\b(products?|services|iphone|mac|ipad|wearables|home|accessories)\b", line_lower):
                return False
        else:
            segment = _question_segment_label(question)
            if segment and segment not in line_lower:
                return False

    if margin_like and "gross margin" in question.lower():
        segment = _question_segment_label(question)
        if segment in {"services", "products"}:
            if segment not in line_lower:
                return False
        elif "gross margin" not in line_lower:
            return False

    return True


def _is_ambiguous_extraction(candidates: list[tuple[float, str, RetrievedChunk, str, str, str, int, str | None]]) -> bool:
    """Detect near-tie top candidates with conflicting numeric values."""
    if len(candidates) < 2:
        return False

    top = candidates[0]
    second = candidates[1]
    score_gap = abs(top[0] - second[0])
    different_values = top[1] != second[1]
    same_chunk = top[2].chunk_id == second[2].chunk_id

    return score_gap < 0.08 and different_values and same_chunk


def _ordered_unique_years(text: str) -> list[str]:
    years: list[str] = []
    for y in re.findall(r"\b20\d{2}\b", text):
        if y not in years:
            years.append(y)
    return years


def _select_value_by_year_header(
    lines: list[str],
    line_index: int,
    values: list[str],
    question_year: str | None,
    quarter_only: bool = False,
) -> tuple[str, dict] | None:
    """Choose the value aligned with the requested year using nearby table headers."""
    if not question_year or len(values) < 2:
        return None

    window_start = max(0, line_index - 8)
    header_window = "\n".join(lines[window_start : line_index + 1])
    years = _ordered_unique_years(header_window)
    if not years or question_year not in years:
        return None

    year_index = years.index(question_year)
    effective_values = values
    mode = "year_header_alignment"
    if quarter_only and len(values) >= 4 and len(years) >= 2:
        # In mixed quarter/YTD rows, prefer the three-month pair first.
        effective_values = values[:2]
        mode = "year_header_alignment_three_months_pair"

    if year_index >= len(effective_values):
        return None

    return effective_values[year_index], {
        "mode": mode,
        "years": years,
        "year_index": year_index,
        "selected_value": effective_values[year_index],
        "value_count": len(values),
    }


def _select_value_by_filing_year(
    values: list[str],
    question_year: str | None,
    filing_date: str | None,
    quarter_only: bool = False,
) -> tuple[str, dict] | None:
    """Fallback mapping using filing year when row headers are missing in a chunk."""
    if not question_year or not filing_date or len(values) < 2:
        return None

    filing_year_match = re.search(r"\b(20\d{2})\b", filing_date)
    if not filing_year_match:
        return None

    filing_year = int(filing_year_match.group(1))
    requested_year = int(question_year)
    index = filing_year - requested_year
    effective_values = values
    mode = "filing_year_offset"
    if quarter_only and len(values) >= 4:
        # In mixed quarter/YTD rows, prefer the three-month pair first.
        effective_values = values[:2]
        mode = "filing_year_offset_three_months_pair"

    if index < 0 or index >= len(effective_values):
        return None

    return effective_values[index], {
        "mode": mode,
        "filing_year": filing_year,
        "requested_year": requested_year,
        "selected_index": index,
        "selected_value": effective_values[index],
        "value_count": len(values),
    }


def _extract_margin_values(line: str) -> list[str]:
    with_percent = [m.group(0).strip() for m in re.finditer(r"\d{1,3}(?:\.\d+)?\s?%", line)]
    if with_percent:
        return with_percent

    decimals = [m.group(0).strip() for m in re.finditer(r"\b\d{1,2}(?:\.\d+)\b", line)]
    filtered: list[str] = []
    for d in decimals:
        try:
            value = float(d)
        except ValueError:
            continue
        if 0.0 <= value <= 100.0:
            filtered.append(d)
    return filtered


def _infer_quarter_from_filing_date(filing_date: str | None) -> str | None:
    if not filing_date:
        return None
    m = re.search(r"\b\d{4}-(\d{2})-\d{2}\b", filing_date)
    if not m:
        return None
    month = int(m.group(1))
    if 1 <= month <= 3:
        return "Q1"
    if 4 <= month <= 6:
        return "Q2"
    if 7 <= month <= 9:
        return "Q3"
    return "Q4"


def _extract_filing_year(filing_date: str | None) -> str | None:
    if not filing_date:
        return None
    m = re.search(r"\b(20\d{2})\b", filing_date)
    return m.group(1) if m else None


def _format_extracted_value(
    value: str,
    *,
    revenue_like: bool,
    annual_question: bool,
    unit_hint: str | None = None,
) -> str:
    cleaned = value.strip()
    cleaned = re.sub(r"\$\s+", "$", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)

    if revenue_like:
        if not cleaned.startswith("$"):
            cleaned = f"${cleaned}"
        if not unit_hint and re.fullmatch(r"\$?\d{1,3}(?:,\d{3})+(?:\.\d+)?", cleaned):
            # SEC statement tables commonly present dollar values in millions.
            unit_hint = "million"
        has_unit = bool(re.search(r"\b(million|billion|bn|m)\b", cleaned, flags=re.IGNORECASE))
        if unit_hint and not has_unit:
            cleaned = f"{cleaned} {unit_hint}"

        if unit_hint == "million":
            num_match = re.search(r"\$?\s*([0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?)", cleaned)
            if num_match:
                try:
                    million_value = float(num_match.group(1).replace(",", ""))
                    billion_value = million_value / 1000.0
                    billion_fmt = f"{billion_value:,.2f}" if billion_value >= 100 else f"{billion_value:,.3f}"
                    million_fmt = num_match.group(1)
                    cleaned = f"${billion_fmt} billion (${million_fmt} million)"
                except ValueError:
                    pass

    return cleaned


def _is_high_confidence_extraction(question: str, evidence: dict | None) -> bool:
    """Conservative guardrail for using deterministic extraction after insufficiency."""
    if not isinstance(evidence, dict):
        return False

    selection_mode = str(evidence.get("selection_mode") or "")
    allowed_modes = {
        "year_header_alignment",
        "year_header_alignment_three_months_pair",
        "filing_year_offset",
        "filing_year_offset_three_months_pair",
    }
    if selection_mode not in allowed_modes:
        return False

    try:
        row_value_count = int(evidence.get("row_value_count") or 0)
    except (TypeError, ValueError):
        row_value_count = 0
    if row_value_count > 4:
        return False

    try:
        top_score = float(evidence.get("top_score"))
    except (TypeError, ValueError):
        top_score = 0.0
    if top_score < 1.0:
        return False

    question_lower = question.lower()
    question_year, question_quarter = _question_period_constraints(question)
    segment = _question_segment_label(question)

    score_gap_raw = evidence.get("score_gap")
    if score_gap_raw is not None:
        try:
            score_gap = float(score_gap_raw)
        except (TypeError, ValueError):
            return False
        try:
            candidate_count = int(evidence.get("candidate_count") or 0)
        except (TypeError, ValueError):
            candidate_count = 0

        min_gap = 0.12
        if (
            selection_mode in {"year_header_alignment_three_months_pair", "filing_year_offset_three_months_pair"}
            and question_year
            and question_quarter
            and candidate_count <= 8
        ):
            # Quarter table rows are often tightly scored across nearby chunks.
            # Allow a smaller margin when explicit year-quarter alignment is used.
            min_gap = 0.015

        if score_gap < min_gap:
            return False

    line_preview = str(evidence.get("line_preview") or "").lower()
    matched_term = str(evidence.get("matched_term") or "").lower()
    filing_type = str(evidence.get("filing_type") or "")
    fiscal_quarter = str(evidence.get("fiscal_quarter") or "").upper()
    inferred_quarter = str(evidence.get("inferred_filing_quarter") or "").upper()

    revenue_like = any(term in question_lower for term in ["revenue", "net sales"])
    if revenue_like:
        if _is_total_revenue_question(question):
            if not any(label in line_preview for label in ["total net sales", "total revenue", "total net revenue"]):
                return False
            if matched_term not in {"total net revenue", "total revenue", "net sales", "total net sales"}:
                return False
        elif segment in {"iphone", "services", "mac", "ipad", "wearables", "products"}:
            if segment not in line_preview:
                return False

    if "gross margin" in question_lower:
        if segment in {"services", "products"} and segment not in line_preview:
            return False
        if segment is None and "gross margin" not in line_preview:
            return False

    if question_quarter:
        if filing_type != "10-Q":
            return False
        if fiscal_quarter and fiscal_quarter != question_quarter:
            return False
        if not fiscal_quarter and inferred_quarter and inferred_quarter != question_quarter:
            return False
    elif question_year and "fiscal year" in question_lower:
        if filing_type != "10-K":
            return False

    return True


def _extract_metric_answer(
    question: str,
    chunks: list[RetrievedChunk],
    scan_limit: int = 20,
) -> tuple[str | None, dict | None]:
    if not _is_direct_lookup_question(question):
        return None, None

    terms = _question_metric_terms(question)
    if not terms:
        return None, None

    question_year, question_quarter = _question_period_constraints(question)
    revenue_like = any(t in terms for t in ["revenue", "total revenue", "total net revenue", "net sales", "total net sales"])
    margin_like = any("margin" in t for t in terms)

    # Metric-specific patterns reduce accidental captures like section numbers.
    money_re = re.compile(
        r"(?:\$\s*\d{1,3}(?:,\d{3})+(?:\.\d+)?(?:\s?(?:million|billion|bn|m))?)"
        r"|(?:\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\s?(?:million|billion|bn|m)\b)"
        r"|(?:\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b)",
        re.IGNORECASE,
    )
    candidates: list[tuple[float, str, RetrievedChunk, str, str, str, int, str | None]] = []

    # When the question asks for a full "fiscal year" figure, only extract
    # from annual filings (10-K) so quarterly 10-Q numbers don't win.
    annual_question = "fiscal year" in question.lower() and not question_quarter

    for c in chunks[:scan_limit]:
        text = c.text
        lower = text.lower()
        if not any(t in lower for t in terms):
            continue
        if c.section_type not in {"financial_statements", "general"}:
            continue
        if annual_question and c.filing_type and c.filing_type != "10-K":
            continue

        inferred_quarter = _infer_quarter_from_filing_date(c.filing_date) if c.filing_type == "10-Q" else None
        if question_quarter:
            if c.fiscal_quarter and str(c.fiscal_quarter).upper() != question_quarter:
                continue
            if not c.fiscal_quarter and inferred_quarter and inferred_quarter != question_quarter:
                continue

        if not (margin_like or revenue_like):
            continue

        chunk_unit_hint: str | None = None
        if re.search(r"\bin billions\b", lower):
            chunk_unit_hint = "billion"
        elif re.search(r"\bin millions\b", lower):
            chunk_unit_hint = "million"

        period_score = 0.0
        if question_year:
            if c.fiscal_year:
                period_score += 0.14 if str(c.fiscal_year) == question_year else -0.10
            elif c.filing_date:
                filing_year_match = re.search(r"\b(20\d{2})\b", c.filing_date)
                if filing_year_match:
                    filing_year = filing_year_match.group(1)
                    if filing_year == question_year:
                        period_score += 0.08
                    elif int(filing_year) - int(question_year) == 1:
                        period_score += 0.04
                    else:
                        period_score -= 0.05
        if question_quarter and c.fiscal_quarter:
            period_score += 0.14 if str(c.fiscal_quarter).upper() == question_quarter else -0.08
        elif question_quarter and inferred_quarter:
            period_score += 0.10 if inferred_quarter == question_quarter else -0.08

        # Prefer row-level extraction for table-like chunks to avoid leaking values
        # from adjacent rows in wide SEC tables.
        lines = text.splitlines()
        for line_index, raw_line in enumerate(lines):
            line = raw_line.strip()
            if not line:
                continue
            line_lower = line.lower()

            matched_term = next((t for t in terms if t in line_lower), None)
            if matched_term is None:
                continue
            if not _line_is_valid_metric_row(line_lower, question, revenue_like, margin_like):
                continue

            if margin_like:
                row_values = _extract_margin_values(line)
            else:
                row_values = [m.group(0).strip() for m in money_re.finditer(line) if m.group(0).strip()]
            if not row_values:
                continue

            selection_candidates: list[tuple[str, float, str]] = []
            selected_by_header = _select_value_by_year_header(
                lines,
                line_index,
                row_values,
                question_year,
                quarter_only=bool(question_quarter),
            )
            if selected_by_header:
                selection_candidates.append((selected_by_header[0], 0.35, selected_by_header[1].get("mode", "year_header_alignment")))
            else:
                selected_by_filing_year = _select_value_by_filing_year(
                    row_values,
                    question_year,
                    c.filing_date,
                    quarter_only=bool(question_quarter),
                )
                if selected_by_filing_year:
                    selection_candidates.append((selected_by_filing_year[0], 0.28, selected_by_filing_year[1].get("mode", "filing_year_offset")))
                else:
                    if question_quarter and len(row_values) > 2:
                        # Ambiguous multi-column quarter row with no reliable year mapping.
                        continue
                    selection_candidates.extend((v, 0.0, "raw_row_value") for v in row_values)

            for val, selection_bonus, selection_mode in selection_candidates:
                if not val:
                    continue

                score = float(c.score)
                if line_lower.startswith("total "):
                    score += 0.25
                if _line_has_total_revenue_label(line_lower):
                    score += 0.20
                if margin_like:
                    score += 0.15
                if revenue_like:
                    score += 0.15
                if c.chunk_kind == "table":
                    score += 0.20
                if question_year and c.fiscal_year and str(c.fiscal_year) == question_year:
                    score += 0.08
                if question_quarter and c.fiscal_quarter and str(c.fiscal_quarter).upper() == question_quarter:
                    score += 0.08
                score += period_score
                score += selection_bonus
                if matched_term in {"total net revenue", "total revenue", "gross margin percentage", "total net sales"}:
                    score += 0.12
                if "fiscal year" in question.lower() and c.filing_type == "10-K":
                    score += 0.15
                if question_quarter and c.filing_type == "10-Q":
                    score += 0.10
                if question_quarter and len(row_values) == 2:
                    score += 0.10
                if question_quarter and len(row_values) > 2:
                    score -= 0.10
                if margin_like and len(row_values) > 2:
                    score -= 0.06

                candidates.append((score, val, c, line[:220], matched_term, selection_mode, len(row_values), chunk_unit_hint))

    if not candidates:
        return None, None

    # Prefer same filing year for quarter-specific questions when available.
    if question_year and question_quarter:
        same_year_candidates = [
            cand for cand in candidates if _extract_filing_year(cand[2].filing_date) == question_year
        ]
        if same_year_candidates:
            candidates = same_year_candidates

    candidates.sort(key=lambda x: x[0], reverse=True)
    if _is_ambiguous_extraction(candidates):
        top = candidates[0]
        second = candidates[1]
        return None, {
            "rejected": "ambiguous_top_candidates",
            "top_value": top[1],
            "second_value": second[1],
            "top_score": round(float(top[0]), 6),
            "second_score": round(float(second[0]), 6),
            "chunk_id": top[2].chunk_id,
        }

    top_score, best_value, best_chunk, best_line, best_term, best_selection_mode, best_row_value_count, best_unit_hint = candidates[0]
    second_score = float(candidates[1][0]) if len(candidates) > 1 else None
    score_gap = (float(top_score) - second_score) if second_score is not None else None

    src = f"{best_chunk.ticker} {best_chunk.filing_type} ({best_chunk.filing_date})" if best_chunk.source_type == "sec_filing" else f"{best_chunk.ticker} Earnings Call ({best_chunk.period})"
    rendered_value = _format_extracted_value(
        best_value,
        revenue_like=revenue_like,
        annual_question=annual_question,
        unit_hint=best_unit_hint,
    )
    answer = f"{rendered_value}. According to {src}."
    evidence = {
        "value": best_value,
        "rendered_value": rendered_value,
        "chunk_id": best_chunk.chunk_id,
        "source": src,
        "matched_term": best_term,
        "line_preview": best_line,
        "section_type": best_chunk.section_type,
        "chunk_kind": best_chunk.chunk_kind,
        "candidate_count": len(candidates),
        "selection_mode": best_selection_mode,
        "row_value_count": best_row_value_count,
        "top_score": round(float(top_score), 6),
        "second_score": round(second_score, 6) if second_score is not None else None,
        "score_gap": round(score_gap, 6) if score_gap is not None else None,
        "unit_hint": best_unit_hint,
        "filing_type": best_chunk.filing_type,
        "filing_date": best_chunk.filing_date,
        "fiscal_year": best_chunk.fiscal_year,
        "fiscal_quarter": best_chunk.fiscal_quarter,
        "inferred_filing_quarter": _infer_quarter_from_filing_date(best_chunk.filing_date),
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
    question_type = str(state.get("question_type", "L1"))
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

    latest_sufficiency = sufficiency_checks[-1] if sufficiency_checks else None
    allow_deterministic_extraction = bool(
        latest_sufficiency
        and latest_sufficiency.get("decision") == "sufficient"
    )

    extraction_scan_limit = 60 if (latest_sufficiency and latest_sufficiency.get("decision") == "insufficient") else 20
    extracted_answer, extracted_evidence = _extract_metric_answer(question, chunks, scan_limit=extraction_scan_limit)

    if allow_deterministic_extraction and extracted_answer:
        diagnostics["generator_extraction"] = {
            "used": True,
            "mode": "sufficient_extraction",
            "evidence": extracted_evidence,
        }
        print("  [generator] Used deterministic metric extraction path")
        return {"answer": extracted_answer, "diagnostics": diagnostics}

    high_confidence_override = bool(
        extracted_answer
        and extracted_evidence
        and latest_sufficiency
        and latest_sufficiency.get("decision") == "insufficient"
        and _is_high_confidence_extraction(question, extracted_evidence)
    )

    if high_confidence_override:
        diagnostics["generator_extraction"] = {
            "used": True,
            "mode": "high_confidence_extraction",
            "override": "latest_sufficiency_insufficient",
            "evidence": extracted_evidence,
        }
        print("  [generator] Used high-confidence extraction override")
        return {"answer": extracted_answer, "diagnostics": diagnostics}

    diagnostics["generator_extraction"] = {"used": False}
    if extracted_evidence:
        diagnostics["generator_extraction"]["rejection"] = extracted_evidence
    if not allow_deterministic_extraction:
        diagnostics["generator_extraction"]["skipped"] = "latest_sufficiency_not_sufficient"

    if latest_sufficiency and latest_sufficiency.get("decision") == "insufficient":
        if question_type == "L1":
            diagnostics["generator_short_circuit"] = "insufficient_context_refusal"
            return {
                "answer": INSUFFICIENT_ANSWER,
                "diagnostics": diagnostics,
            }
        diagnostics["generator_short_circuit"] = "skipped_for_complex_question"

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
    use_partial_prompt = question_type != "L1"
    partial_instruction = ""
    if use_partial_prompt:
        partial_instruction = (
            "\n\nAdditional instruction: This is a complex question with partial evidence. "
            "Answer the supported parts and explicitly identify unsupported parts. "
            "Do not default to full refusal when at least one material part is grounded."
        )

    user_message = f"Source passages:\n\n{context_str}\n\nQuestion: {question}{conflict_instruction}{partial_instruction}"

    try:
        system_prompt = COMPLEX_PARTIAL_SYSTEM_PROMPT if use_partial_prompt else SYSTEM_PROMPT
        answer = llm_call(
            client,
            model=model,
            max_tokens=MAX_TOKENS,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            timeout_sec=45.0,
            max_retries=3,
        )
    except Exception as exc:  # noqa: BLE001
        answer = INSUFFICIENT_ANSWER
        node_errors = dict(state.get("node_errors", {}))
        node_errors["generator"] = str(exc)
        diagnostics = append_node_error_diagnostics(state, "generator", exc)
        print(f"  [generator] WARNING: generation failed ({exc})")
        return {"answer": answer, "node_errors": node_errors, "diagnostics": diagnostics}

    if answer.strip() == INSUFFICIENT_ANSWER:
        diagnostics["generator_refusal_reason"] = "llm_insufficient_context"

    if use_partial_prompt and answer.strip() != INSUFFICIENT_ANSWER:
        unsupported_numbers = _unsupported_answer_numbers(answer, chunks)
        if unsupported_numbers:
            diagnostics["generator_numeric_guard"] = {
                "triggered": True,
                "unsupported_numbers": unsupported_numbers,
            }
            rewrite_user = (
                f"Question:\n{question}\n\n"
                f"Source passages:\n{context_str}\n\n"
                f"Draft answer:\n{answer}\n\n"
                f"Unsupported numeric tokens to remove: {', '.join(unsupported_numbers[:12])}.\n"
                "Rewrite to keep only supported claims and citations."
            )
            try:
                answer = llm_call(
                    client,
                    model=model,
                    max_tokens=MAX_TOKENS,
                    temperature=0.0,
                    messages=[
                        {"role": "system", "content": PARTIAL_REWRITE_SYSTEM_PROMPT},
                        {"role": "user", "content": rewrite_user},
                    ],
                    timeout_sec=45.0,
                    max_retries=2,
                )
            except Exception as exc:  # noqa: BLE001
                diagnostics = append_node_error_diagnostics(state, "generator_rewrite", exc)
                diagnostics["generator_numeric_guard"]["rewrite_error"] = str(exc)
            else:
                diagnostics["generator_numeric_guard"]["rewritten"] = True
                if answer.strip() == INSUFFICIENT_ANSWER:
                    diagnostics["generator_refusal_reason"] = "rewrite_removed_unsupported_numbers"

    print(f"  [generator] Generated {len(answer)} chars using {len(chunks)} chunks")

    return {"answer": answer, "diagnostics": diagnostics}
