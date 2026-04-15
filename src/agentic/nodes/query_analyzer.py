"""
LangGraph node: Query Analyzer

Decomposes the user question into focused sub-questions, extracts metadata (tickers, filing type, source type),
classifies difficulty, and builds SearchFilters for targeted retrieval.
"""

import re
from openai import OpenAI
from src.retrieval import SearchFilters
from src.agentic.nodes.llm_utils import llm_json_call, append_node_error_diagnostics
from src.agentic.nodes.node_schemas import QueryAnalyzerResponse
from src.utils.finance import (
    canonical_period_key,
    normalize_filing_type,
    normalize_fiscal_quarter,
    normalize_section_types,
    normalize_source_type,
    normalize_tickers,
    parse_year_quarter,
)


_SYSTEM = """
You are a query analyzer for financial QA over SEC filings and earnings call transcripts.

Your job:
Given one user question, return exactly one JSON object with fields:
- sub_questions
- tickers
- source_type
- filing_type
- section_types
- fiscal_year
- fiscal_quarter
- period_end_date_from
- period_end_date_to
- question_type

Domain:
- Valid tickers: ["AAPL", "JPM", "F"]
- Company aliases:
    - Apple -> AAPL
    - JPMorgan or JPMorgan Chase -> JPM
    - Ford -> F
- Filing types: "10-K", "10-Q", "8-K"
- Source types: "sec_filing", "ect"

Output schema and rules:
1. Output must be valid JSON only. No markdown, no code fences, no extra text.
2. JSON object keys must be exactly:
    - "sub_questions": array of 1 to 3 non-empty strings
    - "tickers": array of valid ticker strings, or []
    - "source_type": "sec_filing" or "ect" or null
    - "filing_type": "10-K" or "10-Q" or "8-K" or null
        - "section_types": array of section labels, or []
            allowed labels: "risk_factors", "md&a", "financial_statements", "forward_looking_statements", "q_and_a", "prepared_remarks", "general"
        - "fiscal_year": integer year or null
        - "fiscal_quarter": "Q1" | "Q2" | "Q3" | "Q4" | null
        - "period_end_date_from": "YYYY-MM-DD" or null
        - "period_end_date_to": "YYYY-MM-DD" or null
    - "question_type": "L1" or "L2" or "L3" or "L4"
3. Never output unknown keys.
4. Never output values outside allowed enums.
5. If user specifies no ticker, use [].
6. If source is unclear, use null.
7. If filing type is unclear, use null.
8. Do not invent tickers.
9. Use [] for section_types when no section signal exists.

Question type policy:
- L1: direct fact lookup from one period/source.
- L2: cross-period or trend comparison.
- L3: management commentary, tone, guidance, qualitative discussion.
- L4: requires combining multiple sources and/or multiple entities with reasoning.

Sub-question policy:
- If simple fact lookup (L1), use exactly [original question] as one item.
- For L2/L3/L4, decompose into 2 to 3 focused retrieval-friendly sub-questions.
- Keep sub-questions short, specific, and answerable from documents.

Source and filing hints:
- If question asks what management said, call remarks, sentiment, commentary, guidance -> source_type = "ect" (usually L3).
- If question asks for reported metrics, tables, balances, filings language -> source_type = "sec_filing".
- If question explicitly mentions 10-K/10-Q/8-K, set filing_type accordingly; otherwise null.
- If question targets risk factors, set section_types=["risk_factors"].
- If question targets management discussion or analysis, set section_types=["md&a"].
- If question targets prepared remarks or Q&A in calls, map to section_types=["prepared_remarks"] or ["q_and_a"].
- If question explicitly includes a quarter/year, set fiscal_year and fiscal_quarter when possible.

Examples:
Input: What was Apple's Q4 2024 revenue?
Output: {"sub_questions":["What was Apple's Q4 2024 revenue?"],"tickers":["AAPL"],"source_type":"sec_filing","filing_type":"10-Q","section_types":[],"fiscal_year":2024,"fiscal_quarter":"Q4","period_end_date_from":null,"period_end_date_to":null,"question_type":"L1"}

Input: How did Apple's gross margin change from Q3 to Q4 2024?
Output: {"sub_questions":["Apple gross margin Q3 2024","Apple gross margin Q4 2024"],"tickers":["AAPL"],"source_type":"sec_filing","filing_type":"10-Q","section_types":["financial_statements"],"fiscal_year":2024,"fiscal_quarter":null,"period_end_date_from":null,"period_end_date_to":null,"question_type":"L2"}

Input: What did management say about AI investment?
Output: {"sub_questions":["Management commentary on AI investment priorities","Management commentary on expected AI investment impact"],"tickers":[],"source_type":"ect","filing_type":null,"section_types":["prepared_remarks","q_and_a"],"fiscal_year":null,"fiscal_quarter":null,"period_end_date_from":null,"period_end_date_to":null,"question_type":"L3"}

Final instruction:
Return one JSON object only.

Optional user prompt improvement in same file:
Question: {question}
Return exactly one JSON object following the schema above.
"""

_USER = "Question: {question}"


##############################################################################
#                              NODE                                          #
##############################################################################

def _normalize_nullable(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if normalized.lower() in {"", "null", "none", "n/a"}:
        return None
    return normalized


def _infer_temporal_hints(question: str) -> tuple[int | None, str | None]:
    text = question.upper()
    year_match = re.search(r"\b(20\d{2})\b", text)
    fiscal_year = int(year_match.group(1)) if year_match else None
    quarter_match = re.search(r"\bQ([1-4])\b", text)
    fiscal_quarter = f"Q{quarter_match.group(1)}" if quarter_match else None
    parsed_year, parsed_quarter = parse_year_quarter(question)
    fiscal_year = fiscal_year or parsed_year
    fiscal_quarter = fiscal_quarter or parsed_quarter
    return fiscal_year, fiscal_quarter


def _infer_section_hints(question: str) -> list[str] | None:
    text = question.lower()
    hints: list[str] = []
    # Keep section fallback conservative: only infer when section is explicitly requested.
    if "risk factor" in text or "item 1a" in text:
        hints.append("risk_factors")
    if "management discussion" in text or "md&a" in text or "item 7" in text:
        hints.append("md&a")
    if "financial statement" in text or "item 8" in text:
        hints.append("financial_statements")
    if "forward-looking statement" in text:
        hints.append("forward_looking_statements")
    if "q&a" in text or "question and answer" in text:
        hints.append("q_and_a")
    if "prepared remarks" in text:
        hints.append("prepared_remarks")
    return normalize_section_types(hints)


def _infer_ticker_hints(question: str) -> list[str] | None:
    # Try lightweight alias resolution from text when model omits tickers.
    text = question.strip()
    candidates = re.findall(r"\b[A-Za-z][A-Za-z\s\.]{1,30}\b", text)
    # Include full question as candidate for multi-token aliases.
    candidates.append(text)
    return normalize_tickers(candidates)


def _infer_source_hint(question: str) -> str | None:
    q = question.lower()
    # Only infer source from explicit source indicators to avoid false narrowing.
    if any(token in q for token in ["earnings call", "transcript", "prepared remarks", "q&a", "question and answer"]):
        return "ect"
    if any(token in q for token in ["10-k", "10-q", "8-k", "form 10-k", "form 10-q", "form 8-k", "sec filing"]):
        return "sec_filing"
    return None


def _infer_filing_hint(question: str) -> str | None:
    q = question.lower()
    # Infer filing type only from explicit filing references.
    if "10-k" in q or "form 10-k" in q:
        return "10-K"
    if "10-q" in q or "form 10-q" in q:
        return "10-Q"
    if "8-k" in q or "form 8-k" in q:
        return "8-K"
    return None

def query_analyzer_node(state: dict, client: OpenAI, model: str) -> dict:
    """
    Decompose the question and extract retrieval metadata.
    Returns updates for: sub_questions, filters, question_type, original_question
    """
    question = state["question"]

    max_sub_questions = 5

    try:
        parsed = llm_json_call(
            client,
            model=model,
            max_tokens=500,
            schema=QueryAnalyzerResponse,
            required_keys={"sub_questions", "tickers", "source_type", "filing_type", "question_type", "section_types", "fiscal_year", "fiscal_quarter", "period_end_date_from", "period_end_date_to"},
            temperature=0.0,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": _USER.format(question=question)},
            ],
        )

        raw_sub_questions = parsed.sub_questions or [question]
        # Keep decomposition bounded and stable for retrieval budget control.
        sub_questions = [q.strip() for q in raw_sub_questions if isinstance(q, str) and q.strip()]
        if not sub_questions:
            sub_questions = [question]
        deduped: list[str] = []
        seen: set[str] = set()
        for q in sub_questions:
            key = q.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(q)
        sub_questions = deduped[:max_sub_questions]
        tickers       = normalize_tickers(parsed.tickers)
        source_type   = normalize_source_type(_normalize_nullable(parsed.source_type))
        filing_type   = normalize_filing_type(_normalize_nullable(parsed.filing_type))
        section_types = normalize_section_types(parsed.section_types) or []
        fiscal_year   = parsed.fiscal_year
        fiscal_quarter = normalize_fiscal_quarter(parsed.fiscal_quarter)
        canonical_period = parsed.canonical_period
        period_end_date_from = _normalize_nullable(parsed.period_end_date_from)
        period_end_date_to = _normalize_nullable(parsed.period_end_date_to)
        question_type = parsed.question_type or "L1"

        # Deterministic fallbacks when model omits temporal/section hints.
        inferred_year, inferred_quarter = _infer_temporal_hints(question)
        if fiscal_year is None:
            fiscal_year = inferred_year
        if fiscal_quarter is None:
            fiscal_quarter = inferred_quarter
        if tickers is None:
            tickers = _infer_ticker_hints(question)
        if source_type is None:
            source_type = _infer_source_hint(question)
        if filing_type is None:
            filing_type = _infer_filing_hint(question)
        if not section_types:
            section_types = _infer_section_hints(question) or []
        if not canonical_period:
            canonical_period = canonical_period_key(fiscal_year, fiscal_quarter)

        # Only build filters when there's something to filter on
        if tickers or source_type or filing_type or section_types or fiscal_year or fiscal_quarter or canonical_period or period_end_date_from or period_end_date_to:
            filters = SearchFilters(
                tickers=tickers if tickers else None,
                source_type=source_type,
                filing_type=filing_type,
                section_types=section_types if section_types else None,
                fiscal_year=fiscal_year,
                fiscal_quarter=fiscal_quarter,
                canonical_period=canonical_period,
                period_end_date_from=period_end_date_from,
                period_end_date_to=period_end_date_to,
            )
        else:
            filters = None

        print(
            f"  [query_analyzer] type={question_type}, sub_questions={len(sub_questions)}, "
            f"tickers={tickers}, filing_type={filing_type}, section_types={section_types}, "
            f"fiscal_year={fiscal_year}, fiscal_quarter={fiscal_quarter}"
        )

    except Exception as e:
        print(f"  [query_analyzer] WARNING: LLM failed ({e}), using defaults")
        sub_questions = [question]
        filters       = None
        question_type = "L1"
        node_errors = dict(state.get("node_errors", {}))
        node_errors["query_analyzer"] = str(e)
        diagnostics = append_node_error_diagnostics(state, "query_analyzer", e)
        return {
            "sub_questions":     sub_questions,
            "filters":           filters,
            "question_type":     question_type,
            "original_question": question,
            "node_errors":       node_errors,
            "diagnostics":       diagnostics,
        }

    return {
        "sub_questions":     sub_questions,
        "filters":           filters,
        "question_type":     question_type,
        "original_question": question,
    }
