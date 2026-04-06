"""src/nodes/query_analyzer.py

LangGraph node: Query Analyzer

Decomposes the user question into focused sub-questions, extracts
metadata (tickers, filing type, source type), classifies difficulty,
and builds SearchFilters for targeted retrieval.
"""

import json
from openai import OpenAI
from src.retrieval import SearchFilters
from src.nodes.llm_utils import llm_call

##############################################################################
#                              PROMPTS                                       #
##############################################################################

_SYSTEM = """You are a financial question analyst. Given a user question about financial documents (SEC filings and earnings call transcripts), analyze it and output a JSON object.

Available tickers: AAPL, AMD, COST, JPM, F, ELV
Filing types: 10-K (annual), 10-Q (quarterly), 8-K (current events)
Source types: sec_filing, ect (earnings call transcript)

Output a JSON object with these fields:
- sub_questions: list of 1-3 focused sub-questions (use [original_question] for simple L1 questions, decompose into targeted pieces for L2/L3/L4)
- tickers: list of tickers mentioned (e.g. ["AAPL"]), or [] if none specified
- source_type: "sec_filing" | "ect" | null (null = search both)
- filing_type: "10-K" | "10-Q" | "8-K" | null (null = all types)
- question_type: "L1" (direct fact lookup) | "L2" (cross-period comparison) | "L3" (management commentary/sentiment) | "L4" (multi-source reasoning)

Examples:
- "What was Apple's Q4 2024 revenue?" → L1, sub_questions=["What was Apple's Q4 2024 revenue?"], tickers=["AAPL"]
- "How did Apple's gross margin change from Q3 to Q4 2024?" → L2, sub_questions=["Apple gross margin Q3 2024", "Apple gross margin Q4 2024"], tickers=["AAPL"]
- "What did management say about AI investment?" → L3, source_type="ect"
- "Compare Apple and AMD R&D spending in FY2024 with what management said" → L4, tickers=["AAPL","AMD"]

Output ONLY valid JSON. No markdown fences, no prose."""

_USER = "Question: {question}"


##############################################################################
#                              NODE                                          #
##############################################################################

def query_analyzer_node(state: dict, client: OpenAI) -> dict:
    """
    Decompose the question and extract retrieval metadata.

    Returns updates for: sub_questions, filters, question_type, original_question
    """
    question = state["question"]

    try:
        raw = llm_call(
            client,
            model="qwen/qwen-2.5-7b-instruct",
            max_tokens=400,
            temperature=0.0,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": _USER.format(question=question)},
            ],
        )

        # Strip markdown fences if the model adds them
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.rstrip("`").strip()

        parsed = json.loads(raw)

        sub_questions = parsed.get("sub_questions") or [question]
        tickers       = parsed.get("tickers") or None
        source_type   = parsed.get("source_type") or None
        filing_type   = parsed.get("filing_type") or None
        question_type = parsed.get("question_type", "L1")

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

    return {
        "sub_questions":     sub_questions,
        "filters":           filters,
        "question_type":     question_type,
        "original_question": question,
    }
