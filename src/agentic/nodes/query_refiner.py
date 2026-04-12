"""
LangGraph node: Query Refiner

Called when retrieval was insufficient. Rephrases sub-questions to be
broader, relaxes filters, and increments the retrieval_attempts counter
so the graph loops back to the retriever.
"""

from openai import OpenAI
from src.retrieval import SearchFilters
from src.agentic.nodes.llm_utils import llm_call, parse_json_response

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

    try:
        raw = llm_call(
            client,
            model=model,
            max_tokens=200,
            temperature=0.3,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": _USER.format(
                    question=question,
                    sub_questions="\n".join(f"- {q}" for q in prev_sub_questions),
                )},
            ],
        )
        parsed = parse_json_response(raw, required_keys={"sub_questions"})
        new_sub_questions = parsed.get("sub_questions") or [question]

        print(f"  [query_refiner] Attempt {retrieval_attempts + 1}: {len(new_sub_questions)} sub-question(s)")

    except Exception as e:
        print(f"  [query_refiner] WARNING: LLM failed ({e}), falling back to original question")
        new_sub_questions = [question]
        node_errors = dict(state.get("node_errors", {}))
        node_errors["query_refiner"] = str(e)
        return {
            "sub_questions":      new_sub_questions,
            "filters":            SearchFilters(tickers=current_filters.tickers) if current_filters and current_filters.tickers else None,
            "retrieval_attempts": retrieval_attempts + 1,
            "node_errors":        node_errors,
        }

    # Relax filters: keep tickers but drop filing_type and source_type
    # to broaden the search across all document types
    relaxed_filters = None
    if current_filters is not None and current_filters.tickers:
        relaxed_filters = SearchFilters(tickers=current_filters.tickers)

    return {
        "sub_questions":     new_sub_questions,
        "filters":           relaxed_filters,
        "retrieval_attempts": retrieval_attempts + 1,
    }
