"""
LangGraph node: Retriever

Runs hybrid search (Dense + BM25 + RRF) for each sub-question, merges and deduplicates results.
On retry, increases top_k and relaxes filing_type filter.
"""

from dataclasses import replace
import re

from src.retrieval import Retriever, RetrievedChunk, SearchFilters
from src.constants import DEFAULT_TOP_K


def _estimate_source_ref(chunk: RetrievedChunk) -> str | None:
    """Build a canonical source reference path from retrieved chunk metadata."""
    ticker = (chunk.ticker or "").strip()
    if not ticker:
        return None
    if chunk.source_type == "sec_filing":
        if not chunk.filing_type or not chunk.filing_date:
            return None
        compact_date = chunk.filing_date.replace("-", "")
        return f"sec_filings/{ticker}/{chunk.filing_type}/{chunk.filing_type}_{compact_date}.json"
    if not chunk.period:
        return None
    return f"ect/{ticker}/{chunk.period}.json"


def _expand_subquery(sub_q: str) -> str:
    q = sub_q.lower()
    extras: list[str] = []
    if "revenue" in q:
        extras.extend(["net sales", "total net sales"])
    if "gross margin" in q:
        extras.extend(["gross margin percentage", "products gross margin", "services gross margin"])
    if "operating margin" in q:
        extras.extend(["operating income", "operating expenses"])
    if not extras:
        return sub_q
    return f"{sub_q} | related terms: {', '.join(dict.fromkeys(extras))}"


def _is_low_information_chunk(chunk: RetrievedChunk) -> bool:
    text = (chunk.text or "").strip()
    if len(text) < 40:
        return True

    one_line = re.sub(r"\s+", " ", text)
    # Common page-header fragments in SEC docs add noise to top-k results.
    if re.fullmatch(r"[A-Za-z0-9 .,'()\-]+\|\s*Q[1-4]\s+\d{4}\s+Form\s+10-[KQ]\s*\|\s*\d{1,3}", one_line):
        return True
    return False


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
    diagnostics        = dict(state.get("diagnostics", {}))

    # On retry: increase top_k to cast a wider net
    top_k = DEFAULT_TOP_K + (retrieval_attempts * 5)

    # Retry relaxation policy (progressive):
    # 1) drop filing_type, 2) drop section_types, 3) drop source_type only when ticker is not fixed.
    relaxed_fields: list[str] = []
    if retrieval_attempts > 0 and filters is not None:
        if retrieval_attempts >= 1 and filters.filing_type:
            filters = replace(filters, filing_type=None)
            relaxed_fields.append("filing_type")
        if retrieval_attempts >= 2 and filters.section_types:
            filters = replace(filters, section_types=None)
            relaxed_fields.append("section_types")
        if retrieval_attempts >= 3 and filters.source_type and not filters.tickers:
            filters = replace(filters, source_type=None)
            relaxed_fields.append("source_type")

    if relaxed_fields:
        print(f"  [retriever] Retry {retrieval_attempts}: top_k={top_k}, relaxed={relaxed_fields}, filters={filters}")
    else:
        print(f"  [retriever] top_k={top_k}, filters={filters}")

    # Run search for each sub-question, deduplicate by chunk_id
    seen_ids: set[str] = set()
    all_chunks: list[RetrievedChunk] = []

    for sub_q in sub_questions:
        expanded_q = _expand_subquery(sub_q)
        chunks = retriever.search(expanded_q, k=top_k, filters=filters)
        for chunk in chunks:
            if chunk.chunk_id not in seen_ids:
                seen_ids.add(chunk.chunk_id)
                all_chunks.append(chunk)

    all_chunks = [c for c in all_chunks if not _is_low_information_chunk(c)]

    # Sort merged results by RRF score descending
    all_chunks.sort(key=lambda c: c.score, reverse=True)

    # Cap total chunks to avoid overwhelming the generator context
    max_chunks = top_k * len(sub_questions)
    all_chunks = all_chunks[:max_chunks]

    print(f"  [retriever] {len(all_chunks)} unique chunks from {len(sub_questions)} sub-question(s)")

    # Track retrieval overlap across attempts so routing can stop low-yield retries.
    history = list(diagnostics.get("retrieval_history", []))
    current_ids = [c.chunk_id for c in all_chunks]
    overlap_prev = None
    if history:
        prev_ids = set(history[-1].get("chunk_ids", []))
        cur_ids = set(current_ids)
        if prev_ids and cur_ids:
            overlap_prev = len(prev_ids & cur_ids) / max(1, len(cur_ids))

    history.append({
        "attempt": retrieval_attempts,
        "top_k": top_k,
        "chunk_count": len(all_chunks),
        "chunk_ids": current_ids,
        "source_refs": sorted({ref for ref in (_estimate_source_ref(c) for c in all_chunks) if ref}),
        "overlap_with_prev": overlap_prev,
    })
    diagnostics["retrieval_history"] = history

    if overlap_prev is not None:
        print(f"  [retriever] overlap with previous attempt: {overlap_prev:.2f}")

    return {"retrieved_chunks": all_chunks, "diagnostics": diagnostics}
