"""src/nodes/retriever_node.py

LangGraph node: Retriever

Runs hybrid search (Dense + BM25 + RRF) for each sub-question,
merges and deduplicates results. On retry, increases top_k and
relaxes filing_type filter.
"""

from src.retrieval import Retriever, RetrievedChunk, SearchFilters

DEFAULT_TOP_K = 5


##############################################################################
#                              NODE                                          #
##############################################################################

def retriever_node(state: dict, retriever: Retriever) -> dict:
    """
    Retrieve chunks for all sub-questions, merge by chunk_id (dedup).

    Returns updates for: retrieved_chunks
    """
    sub_questions      = state.get("sub_questions", [state["question"]])
    filters            = state.get("filters")
    retrieval_attempts = state.get("retrieval_attempts", 0)

    # On retry: increase top_k to cast a wider net
    top_k = DEFAULT_TOP_K + (retrieval_attempts * 5)

    # On retry: relax the filing_type filter (keep tickers, drop filing type)
    if retrieval_attempts > 0 and filters is not None and filters.filing_type:
        filters = SearchFilters(
            tickers=filters.tickers,
            source_type=filters.source_type,
            filing_type=None,
        )
        print(f"  [retriever] Retry {retrieval_attempts}: top_k={top_k}, filing_type filter relaxed")
    else:
        print(f"  [retriever] top_k={top_k}, filters={filters}")

    # Run search for each sub-question, deduplicate by chunk_id
    seen_ids: set[str] = set()
    all_chunks: list[RetrievedChunk] = []

    for sub_q in sub_questions:
        chunks = retriever.search(sub_q, k=top_k, filters=filters)
        for chunk in chunks:
            if chunk.chunk_id not in seen_ids:
                seen_ids.add(chunk.chunk_id)
                all_chunks.append(chunk)

    # Sort merged results by RRF score descending
    all_chunks.sort(key=lambda c: c.score, reverse=True)

    # Cap total chunks to avoid overwhelming the generator context
    max_chunks = top_k * len(sub_questions)
    all_chunks = all_chunks[:max_chunks]

    print(f"  [retriever] {len(all_chunks)} unique chunks from {len(sub_questions)} sub-question(s)")

    return {"retrieved_chunks": all_chunks}
