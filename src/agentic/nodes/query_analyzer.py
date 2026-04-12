"""
LangGraph node: Query Analyzer

Decomposes the user question into focused sub-questions, extracts metadata (tickers, filing type, source type),
classifies difficulty, and builds SearchFilters for targeted retrieval.
"""

from openai import OpenAI
from src.retrieval import SearchFilters
from src.agentic.nodes.llm_utils import llm_json_call, append_node_error_diagnostics
from src.agentic.nodes.node_schemas import QueryAnalyzerResponse

##############################################################################
#                              PROMPTS                                       #
##############################################################################

# _SYSTEM = """You are a financial analyst. Given a question about financial documents (SEC filings and earnings call transcripts), analyze it and output a JSON object.

# Available tickers: AAPL, AMD, COST, JPM, F, ELV
# Filing types: 10-K (annual), 10-Q (quarterly), 8-K (current events)
# Source types: sec_filing, ect (earnings call transcript)

# Output a JSON object with these fields:
# - sub_questions: list of 1-3 focused sub-questions (use [original_question] for simple L1 questions, decompose into targeted pieces for L2/L3/L4)
# - tickers: list of relevant tickers (e.g. ["AAPL"]), or [] if none specified
# - source_type: "sec_filing" | "ect" | null (null = search both)
# - filing_type: "10-K" | "10-Q" | "8-K" | null (null = all types)
# - question_type: "L1" (direct fact lookup) | "L2" (cross-period comparison) | "L3" (management commentary/sentiment) | "L4" (multi-source reasoning)

# Examples:
# - "What was Apple's Q4 2024 revenue?" → L1, sub_questions=["What was Apple's Q4 2024 revenue?"], tickers=["AAPL"]
# - "How did Apple's gross margin change from Q3 to Q4 2024?" → L2, sub_questions=["Apple gross margin Q3 2024", "Apple gross margin Q4 2024"], tickers=["AAPL"]
# - "What did management say about AI investment?" → L3, source_type="ect"
# - "Compare Apple and AMD R&D spending in FY2024 with what management said" → L4, tickers=["AAPL","AMD"]

# Output ONLY valid JSON. No markdown fences, no prose."""

_SYSTEM = """
You are a query analyzer for financial QA over SEC filings and earnings call transcripts.

Your job:
Given one user question, return exactly one JSON object with fields:
- sub_questions
- tickers
- source_type
- filing_type
- question_type

Domain:
- Valid tickers: ["AAPL","AMD","COST","JPM","F","ELV"]
- Company aliases:
    - Apple -> AAPL
    - AMD or Advanced Micro Devices -> AMD
    - Costco -> COST
    - JPMorgan or JPMorgan Chase -> JPM
    - Ford -> F
    - Elevance or Elevance Health -> ELV
- Filing types: "10-K", "10-Q", "8-K"
- Source types: "sec_filing", "ect"

Output schema and rules:
1. Output must be valid JSON only. No markdown, no code fences, no extra text.
2. JSON object keys must be exactly:
    - "sub_questions": array of 1 to 3 non-empty strings
    - "tickers": array of valid ticker strings, or []
    - "source_type": "sec_filing" or "ect" or null
    - "filing_type": "10-K" or "10-Q" or "8-K" or null
    - "question_type": "L1" or "L2" or "L3" or "L4"
3. Never output unknown keys.
4. Never output values outside allowed enums.
5. If user specifies no ticker, use [].
6. If source is unclear, use null.
7. If filing type is unclear, use null.
8. Do not invent tickers.

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

Examples:
Input: What was Apple's Q4 2024 revenue?
Output: {"sub_questions":["What was Apple's Q4 2024 revenue?"],"tickers":["AAPL"],"source_type":"sec_filing","filing_type":"10-Q","question_type":"L1"}

Input: How did Apple's gross margin change from Q3 to Q4 2024?
Output: {"sub_questions":["Apple gross margin Q3 2024","Apple gross margin Q4 2024"],"tickers":["AAPL"],"source_type":"sec_filing","filing_type":"10-Q","question_type":"L2"}

Input: What did management say about AI investment?
Output: {"sub_questions":["Management commentary on AI investment priorities","Management commentary on expected AI investment impact"],"tickers":[],"source_type":"ect","filing_type":null,"question_type":"L3"}

Input: Compare Apple and AMD R&D spending in FY2024 with what management said.
Output: {"sub_questions":["Apple R&D spending in FY2024","AMD R&D spending in FY2024","Management commentary on R&D strategy for Apple and AMD"],"tickers":["AAPL","AMD"],"source_type":null,"filing_type":"10-K","question_type":"L4"}

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
            required_keys={"sub_questions", "tickers", "source_type", "filing_type", "question_type"},
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
        tickers_raw   = parsed.tickers or None
        tickers       = [t.strip() for t in tickers_raw if isinstance(t, str) and t.strip().lower() not in {"null", "none"}] if tickers_raw else None
        source_type   = _normalize_nullable(parsed.source_type)
        filing_type   = _normalize_nullable(parsed.filing_type)
        question_type = parsed.question_type or "L1"

        # Only build filters when there's something to filter on
        if tickers or source_type or filing_type:
            filters = SearchFilters(
                tickers=tickers if tickers else None,
                source_type=source_type,
                filing_type=filing_type,
            )
        else:
            filters = None

        print(f"  [query_analyzer] type={question_type}, sub_questions={len(sub_questions)}, tickers={tickers}, filing_type={filing_type}")

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
