"""
LangGraph node: Query Refiner

Called when retrieval was insufficient. Rephrases sub-questions to be broader, relaxes filters, 
and increments the retrieval_attempts counter so the graph loops back to the retriever.
"""

from openai import OpenAI
from src.retrieval import SearchFilters
from src.agentic.nodes.llm_utils import llm_json_call, append_node_error_diagnostics
from src.agentic.nodes.node_schemas import QueryRefinerResponse
from dataclasses import replace

##############################################################################
#                              PROMPTS                                       #
##############################################################################

_SYSTEM = """You are refining financial search queries that failed to retrieve sufficient information.

Produce broader, more general reformulations of the given sub-questions. Use alternative terminology, remove overly specific time constraints, and consider synonyms.

Output a JSON object:
- sub_questions: list of 1-3 broader sub-questions

Output ONLY valid JSON. No markdown fences, no prose."""

_USER = """ORIGINAL QUESTION: {question}

PREVIOUS SUB-QUESTIONS (insufficient results):
{sub_questions}

Reformulate these as broader sub-questions that are more likely to match relevant passages."""


##############################################################################
#                              NODE                                          #
##############################################################################

def query_refiner_node(state: dict, client: OpenAI, model: str) -> dict:
    """
    Broaden sub-questions and relax filters for a retry retrieval pass.

    Returns updates for: sub_questions, filters, retrieval_attempts
    """
    question           = state["original_question"]
    prev_sub_questions = state.get("sub_questions", [question])
    retrieval_attempts = state.get("retrieval_attempts", 0)
    current_filters    = state.get("filters")

    max_sub_questions = 5

    try:
        parsed = llm_json_call(
            client,
            model=model,
            max_tokens=500,
            schema=QueryRefinerResponse,
            required_keys={"sub_questions"},
            temperature=0.3,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": _USER.format(
                    question=question,
                    sub_questions="\n".join(f"- {q}" for q in prev_sub_questions),
                )},
            ],
        )
        raw_sub_questions = parsed.sub_questions or [question]
        new_sub_questions = [q.strip() for q in raw_sub_questions if isinstance(q, str) and q.strip()]
        if not new_sub_questions:
            new_sub_questions = [question]
        deduped: list[str] = []
        seen: set[str] = set()
        for q in new_sub_questions:
            key = q.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(q)
        new_sub_questions = deduped[:max_sub_questions]

        print(f"  [query_refiner] Attempt {retrieval_attempts + 1}: {len(new_sub_questions)} sub-question(s)")

    except Exception as e:
        print(f"  [query_refiner] WARNING: LLM failed ({e}), falling back to original question")
        new_sub_questions = [question]
        node_errors = dict(state.get("node_errors", {}))
        node_errors["query_refiner"] = str(e)
        diagnostics = append_node_error_diagnostics(state, "query_refiner", e)
        return {
            "sub_questions":      new_sub_questions,
            "filters":            SearchFilters(tickers=current_filters.tickers) if current_filters and current_filters.tickers else None,
            "retrieval_attempts": retrieval_attempts + 1,
            "node_errors":        node_errors,
            "diagnostics":        diagnostics,
        }

    # Progressive relaxation by retry stage to avoid over-broad jumps.
    # Stage 1: drop filing_type, keep source_type and tickers.
    # Stage 2: drop filing_type and source_type, keep tickers.
    # Stage 3+: fully unfiltered.
    next_attempt = retrieval_attempts + 1
    relaxed_filters = None
    if current_filters is not None:
        if next_attempt == 1:
            relaxed_filters = replace(current_filters, filing_type=None)
        elif next_attempt == 2:
            relaxed_filters = replace(current_filters, filing_type=None, source_type=None)
        else:
            relaxed_filters = None

    print(f"  [query_refiner] Relaxed filters for attempt {next_attempt}: {relaxed_filters}")

    return {
        "sub_questions":     new_sub_questions,
        "filters":           relaxed_filters,
        "retrieval_attempts": next_attempt,
    }
