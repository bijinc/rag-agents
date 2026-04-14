import chromadb
import re
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from dataclasses import dataclass
from src.constants import EMBEDDING_MODEL, COLLECTION_NAME, CHROMA_DB_PATH, DEFAULT_TOP_K
from src.finance_domain import canonical_period_key, parse_year_quarter

RRF_K = 60       # constant from the original RRF paper (Cormack et al. 2009)
TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?")
QUERY_YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
QUERY_QUARTER_PATTERN = re.compile(r"\b(?:q([1-4])|quarter\s*([1-4])|([1-4])q)\b", re.IGNORECASE)

FINANCE_SYNONYMS: dict[str, list[str]] = {
    "revenue": ["sales", "topline", "turnover"],
    "guidance": ["outlook", "forecast", "projection"],
    "margin": ["gross", "operating", "profitability"],
    "capex": ["capital", "expenditure", "investment"],
    "opex": ["operating", "expenses", "costs"],
    "eps": ["earnings", "share"],
    "cash": ["liquidity", "flow", "working", "capital"],
    "debt": ["leverage", "borrowings", "liabilities"],
    "buyback": ["repurchase", "shares"],
    "risk": ["headwind", "uncertainty", "exposure"],
}

SOURCE_HINTS = {
    "sec_filing": ["10-k", "10-q", "8-k", "filing", "item", "sec"],
    "ect": ["transcript", "earnings", "call", "operator", "q&a", "remarks"],
}

SECTION_HINTS: dict[str, list[str]] = {
    "risk_factors": ["risk", "uncertainty", "exposure", "headwind"],
    "md&a": ["management", "discussion", "analysis", "outlook"],
    "financial_statements": ["balance", "income", "cash", "statement", "gaap"],
    "forward_looking_statements": ["forward", "guidance", "forecast", "outlook"],
    "q_and_a": ["question", "answer", "analyst", "q&a"],
    "prepared_remarks": ["prepared", "remarks", "operator"],
    "general": [],
}

CHUNK_KIND_HINTS = {
    "table": ["table", "breakdown", "%", "percent", "reconciliation"],
    "semantic": ["trend", "theme", "drivers"],
    "prose": ["commentary", "discussion", "narrative"],
}


def _regex_tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall((text or "").lower())


def _contains_any(query_lc: str, phrases: list[str]) -> bool:
    return any(p in query_lc for p in phrases)


def _extract_temporal_hints(query: str) -> tuple[set[str], set[str]]:
    q = query.lower()
    years = {m.group(0) for m in QUERY_YEAR_PATTERN.finditer(q)}
    quarters: set[str] = set()
    for m in QUERY_QUARTER_PATTERN.finditer(q):
        qn = m.group(1) or m.group(2) or m.group(3)
        if qn:
            quarters.add(str(qn))
    return years, quarters


def _period_parts_from_meta(meta: dict) -> tuple[int | None, str | None]:
    return parse_year_quarter(meta.get("period"))

@dataclass
class SearchFilters:
    """Optional filters to scope retrieval to a subset of the corpus."""
    tickers: list[str] | None = None
    source_type: str | None = None    # "sec_filing" | "ect"
    filing_type: str | None = None    # "10-K" | "10-Q" | "8-K"
    section_types: list[str] | None = None
    date_from: str | None = None      # "YYYY-MM-DD" — SEC filings lower bound
    date_to: str | None = None        # "YYYY-MM-DD" — SEC filings upper bound
    period_end_date_from: str | None = None
    period_end_date_to: str | None = None
    year: int | None = None           # ECT year
    quarter: int | None = None        # ECT quarter (1–4)
    fiscal_year: int | None = None
    fiscal_quarter: str | None = None
    canonical_period: str | None = None  # normalized format: YYYY-Qn


@dataclass
class RetrievedChunk:
    """A single retrieved chunk with its relevance score"""
    chunk_id: str
    text: str
    score: float
    ticker: str
    company_name: str
    source_type: str                    # "sec_filing" | "ect"
    filing_type: str | None = None      # SEC only
    filing_date: str | None = None      # SEC only
    year: str | None = None             # ECT only
    quarter: str | None = None          # ECT only
    period: str | None = None           # ECT only
    period_end_date: str | None = None
    fiscal_year: str | None = None
    fiscal_quarter: str | None = None
    canonical_period: str | None = None
    fiscal_period_sort: str | None = None
    section_type: str | None = None
    chunk_kind: str | None = None
    speaker_count: str | None = None
    doc_id: str | None = None


class Retriever:
    """
    Hybrid retriever combining:
      - Dense search  : ChromaDB cosine similarity on all-MiniLM-L6-v2 embeddings
      - Sparse search : BM25Okapi over an in-memory copy of the corpus
      - Fusion        : Reciprocal Rank Fusion (RRF) as the primary entry point

    The BM25 corpus is loaded once at init from ChromaDB so both paths operate on exactly the same set of chunks.
    """

    def __init__(self):
        print("Initializing Retriever...")

        self._client = chromadb.PersistentClient(CHROMA_DB_PATH)
        self._collection = self._client.get_collection(COLLECTION_NAME)
        print(f"ChromaDB: '{COLLECTION_NAME}' ({self._collection.count()} chunks)")

        self._embed_model = SentenceTransformer(EMBEDDING_MODEL)

        self._load_bm25_corpus()
        print(f"BM25 corpus: {len(self._corpus_ids)} chunks\n")

    def _load_bm25_corpus(self):
        """Pull every chunk out of ChromaDB and pre-tokenize for BM25."""
        total = self._collection.count()
        page_size = 2000

        self._corpus_ids = []
        self._corpus_texts = []
        self._corpus_metadatas = []

        for offset in range(0, total, page_size):
            page = self._collection.get(
                include=["documents", "metadatas"],
                limit=min(page_size, total - offset),
                offset=offset,
            )
            self._corpus_ids.extend(page["ids"])
            self._corpus_texts.extend(page["documents"])
            self._corpus_metadatas.extend(page["metadatas"])
        self._tokenized_corpus = [_regex_tokenize(text) for text in self._corpus_texts]

    def _expand_bm25_query(self, query: str, filters: SearchFilters | None) -> list[str]:
        tokens = _regex_tokenize(query)
        expanded = list(tokens)
        seen = set(tokens)
        ql = query.lower()

        for token in tokens:
            for syn in FINANCE_SYNONYMS.get(token, []):
                for syn_tok in _regex_tokenize(syn):
                    if syn_tok not in seen:
                        expanded.append(syn_tok)
                        seen.add(syn_tok)

        if filters and filters.source_type in SOURCE_HINTS:
            source_terms = SOURCE_HINTS[filters.source_type]
        elif _contains_any(ql, SOURCE_HINTS["ect"]):
            source_terms = SOURCE_HINTS["ect"]
        elif _contains_any(ql, SOURCE_HINTS["sec_filing"]):
            source_terms = SOURCE_HINTS["sec_filing"]
        else:
            source_terms = []

        for term in source_terms:
            for term_tok in _regex_tokenize(term):
                if term_tok not in seen:
                    expanded.append(term_tok)
                    seen.add(term_tok)

        return expanded

    def _query_metadata_boost(self, chunk: RetrievedChunk, query: str, filters: SearchFilters | None) -> float:
        boost = 0.0
        ql = query.lower()

        expected_source = filters.source_type if filters and filters.source_type else None
        if expected_source is None:
            if _contains_any(ql, SOURCE_HINTS["ect"]):
                expected_source = "ect"
            elif _contains_any(ql, SOURCE_HINTS["sec_filing"]):
                expected_source = "sec_filing"
        if expected_source and chunk.source_type == expected_source:
            boost += 0.04

        # Section type boost: both query-keyword hinting and explicit filter matching
        if chunk.section_type:
            section_terms = SECTION_HINTS.get(chunk.section_type, [])
            if section_terms and _contains_any(ql, section_terms):
                boost += 0.05
            # Boost chunks matching the requested section_types from filters
            if filters and filters.section_types and chunk.section_type in filters.section_types:
                boost += 0.06

        if chunk.chunk_kind:
            kind_terms = CHUNK_KIND_HINTS.get(chunk.chunk_kind, [])
            if kind_terms and _contains_any(ql, kind_terms):
                boost += 0.02

        years, quarters = _extract_temporal_hints(query)
        if years:
            filing_year = (chunk.filing_date or "")[:4]
            if filing_year in years or (chunk.period and any(y in chunk.period for y in years)):
                boost += 0.03

        if quarters:
            period_q = None
            if chunk.quarter:
                period_q = str(chunk.quarter)
            elif chunk.period:
                _, pq = parse_year_quarter(chunk.period)
                period_q = str(pq) if pq else None
            if period_q and period_q in quarters:
                boost += 0.02

        return boost

    def _document_key(self, chunk: RetrievedChunk) -> str:
        if chunk.doc_id:
            return chunk.doc_id
        if "_" in chunk.chunk_id:
            return chunk.chunk_id.rsplit("_", 1)[0]
        return chunk.chunk_id


    def _build_chroma_filter(self, filters: SearchFilters | None) -> dict | None:
        """
        Translate SearchFilters into a ChromaDB 'where' clause.
        Returns None when no filter is needed so the caller can skip the kwarg.

        NOTE: section_types is intentionally excluded from hard filtering here
        because ~90% of SEC and ~99% of ECT chunks are classified as "general".
        Hard-filtering on section_type excludes most relevant chunks. Instead,
        section_type matching is handled as a soft boost in _query_metadata_boost.
        Similarly, filing_type is excluded when source_type is 'ect' since ECT
        chunks don't have a filing_type field.
        """
        if filters is None:
            return None

        clauses = []

        if filters.tickers:
            if len(filters.tickers) == 1:
                clauses.append({"ticker": {"$eq": filters.tickers[0]}})
            else:
                clauses.append({"ticker": {"$in": filters.tickers}})

        if filters.source_type:
            clauses.append({"source_type": {"$eq": filters.source_type}})

        # Only hard-filter on filing_type for SEC filings — ECT chunks don't
        # have this field, so applying it would return 0 results.
        if filters.filing_type and filters.source_type != "ect":
            clauses.append({"filing_type": {"$eq": filters.filing_type}})

        # section_types: soft boost only (see _query_metadata_boost), not hard filter

        # Temporal filtering: fiscal_year narrows to the right annual period
        if filters.fiscal_year is not None:
            clauses.append({"fiscal_year": {"$eq": str(filters.fiscal_year)}})

        if filters.date_from:
            clauses.append({"filing_date": {"$gte": filters.date_from}})

        if filters.date_to:
            clauses.append({"filing_date": {"$lte": filters.date_to}})

        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def _metadata_matches(self, meta: dict, filters: SearchFilters | None) -> bool:
        """In-memory metadata predicate used to pre-filter the BM25 corpus."""
        if filters is None:
            return True
        if filters.tickers and meta.get("ticker") not in filters.tickers:
            return False
        if filters.source_type and meta.get("source_type") != filters.source_type:
            return False
        # Only hard-filter filing_type for SEC filings; ECT chunks lack this field.
        if filters.filing_type and meta.get("source_type") != "ect" and meta.get("filing_type") != filters.filing_type:
            return False
        # section_types: soft boost only — not a hard filter (see _query_metadata_boost)
        if filters.date_from and meta.get("filing_date", "") < filters.date_from:
            return False
        if filters.date_to and meta.get("filing_date", "") > filters.date_to:
            return False

        parsed_year, parsed_quarter = _period_parts_from_meta(meta)

        if filters.year is not None and parsed_year is not None and parsed_year != filters.year:
            return False
        if filters.quarter is not None and parsed_quarter is not None and parsed_quarter != f"Q{filters.quarter}":
            return False

        if filters.fiscal_year is not None:
            meta_fiscal_year = meta.get("fiscal_year")
            if not meta_fiscal_year and parsed_year is not None:
                meta_fiscal_year = str(parsed_year)
            if meta_fiscal_year and str(meta_fiscal_year) != str(filters.fiscal_year):
                return False

        if filters.fiscal_quarter is not None:
            meta_fiscal_quarter = meta.get("fiscal_quarter")
            if not meta_fiscal_quarter and parsed_quarter is not None:
                meta_fiscal_quarter = parsed_quarter
            if meta_fiscal_quarter and str(meta_fiscal_quarter) != str(filters.fiscal_quarter):
                return False

        if filters.canonical_period is not None:
            meta_canonical = meta.get("canonical_period")
            if not meta_canonical:
                fy = meta.get("fiscal_year")
                fq = meta.get("fiscal_quarter")
                meta_canonical = canonical_period_key(fy, fq)
                if not meta_canonical:
                    meta_canonical = canonical_period_key(parsed_year, parsed_quarter)
            if meta_canonical and meta_canonical != str(filters.canonical_period):
                return False

        return True


    def dense_search(self,query: str,k: int = DEFAULT_TOP_K, filters: SearchFilters | None = None,) -> list[RetrievedChunk]:
        """
        Top-k retrieval by embedding similarity.
        ChromaDB stores L2 distances; we convert to a score via 1/(1+d).
        """
        n = min(k, self._collection.count())
        if n == 0:
            return []

        query_embedding = self._embed_model.encode(query).tolist()
        where = self._build_chroma_filter(filters)

        query_kwargs: dict = dict(
            query_embeddings=[query_embedding],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        if where:
            query_kwargs["where"] = where

        result = self._collection.query(**query_kwargs)

        dense_chunks = [
            _make_chunk(cid, doc, 1.0 / (1.0 + dist), meta)
            for cid, doc, dist, meta in zip(
                result["ids"][0],
                result["documents"][0],
                result["distances"][0],
                result["metadatas"][0],
            )
        ]

        if filters is None:
            return dense_chunks[:k]

        filtered = [c for c in dense_chunks if self._metadata_matches(c.__dict__, filters)]
        return filtered[:k]


    def bm25_search(self,query: str,k: int = DEFAULT_TOP_K,filters: SearchFilters | None = None,) -> list[RetrievedChunk]:
        """
        Top-k retrieval by BM25Okapi keyword scoring.
        The corpus is filtered in-memory before scoring so that BM25 sees the same slice of documents as the dense search.
        """
        indices = [
            i for i, meta in enumerate(self._corpus_metadatas)
            if self._metadata_matches(meta, filters)
        ]
        if not indices:
            return []

        bm25 = BM25Okapi([self._tokenized_corpus[i] for i in indices])
        expanded_query_tokens = self._expand_bm25_query(query, filters)
        if not expanded_query_tokens:
            return []
        scores = bm25.get_scores(expanded_query_tokens)

        top_local = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        return [
            _make_chunk(
                self._corpus_ids[indices[li]],
                self._corpus_texts[indices[li]],
                float(scores[li]),
                self._corpus_metadatas[indices[li]],
            )
            for li in top_local
        ]


    # Financial metric keywords → ChromaDB where_document search terms
    _METRIC_KEYWORDS: dict[str, list[str]] = {
        "revenue":     ["Total net sales", "net sales", "total revenue"],
        "net income":  ["Net income", "net income"],
        "gross margin":["gross margin", "Gross margin"],
        "eps":         ["earnings per share", "Diluted earnings per share"],
        "operating":   ["Total operating expenses", "operating income"],
        "ebit":        ["EBIT", "operating income", "adjusted EBIT"],
        "cash flow":   ["Cash generated by operating", "free cash flow"],
        "dividend":    ["dividends", "per share"],
        "buyback":     ["repurchase", "treasury stock"],
        "r&d":         ["Research and development"],
        "iphone":      ["iPhone"],
        "services":    ["Services"],
        "ipad":        ["iPad"],
        "mac":         ["Mac"],
    }

    def _keyword_table_search(
        self, query: str, k: int, filters: SearchFilters | None = None,
    ) -> list[RetrievedChunk]:
        """Search for table chunks containing financial metric keywords.

        This third retrieval leg compensates for embedding models that rank
        prose *about* revenue higher than tables *containing* revenue numbers.
        Uses ChromaDB's where_document $contains filter for exact substring
        matching, which directly finds "Total net sales  $391,035".
        """
        ql = query.lower()
        search_terms: list[str] = []
        for keyword, terms in self._METRIC_KEYWORDS.items():
            if keyword in ql:
                search_terms.extend(terms)
                break  # one set of terms per query is enough

        if not search_terms:
            return []

        chroma_filter = self._build_chroma_filter(filters)
        all_chunks: list[RetrievedChunk] = []
        seen_ids: set[str] = set()

        for term in search_terms[:3]:  # limit to avoid excessive queries
            try:
                kwargs: dict = {"query_texts": [query], "n_results": min(k, 20)}
                if chroma_filter:
                    kwargs["where"] = chroma_filter
                kwargs["where_document"] = {"$contains": term}
                results = self._collection.query(**kwargs)

                if results and results["ids"] and results["ids"][0]:
                    for cid, doc, meta, dist in zip(
                        results["ids"][0],
                        results["documents"][0],
                        results["metadatas"][0],
                        results["distances"][0],
                    ):
                        if cid not in seen_ids:
                            seen_ids.add(cid)
                            score = 1.0 / (1.0 + dist)
                            all_chunks.append(_make_chunk(cid, doc, score, meta))
            except Exception:
                continue

        all_chunks.sort(key=lambda c: c.score, reverse=True)
        return all_chunks[:k]

    def search(self,query: str, k: int = DEFAULT_TOP_K, filters: SearchFilters | None = None,) -> list[RetrievedChunk]:
        """
        Hybrid search via Reciprocal Rank Fusion.

        Each system contributes 2*k candidates; a chunk that ranks highly in
        both lists receives the highest fused score.  RRF score for document d:

            score(d) = Σ  1 / (RRF_K + rank_i(d))
                      lists

        RRF_K=60 is the standard value from Cormack et al. (2009).
        """
        if k <= 0:
            return []

        pool_k = min(self._collection.count(), max(k * 4, k + 20))
        dense_results = self.dense_search(query, k=pool_k, filters=filters)
        bm25_results = self.bm25_search(query, k=pool_k, filters=filters)

        # Third retrieval leg: keyword-based document search for financial
        # metric terms.  This catches table chunks that contain the actual
        # numbers (e.g. "$391,035") which embedding models rank poorly.
        keyword_results = self._keyword_table_search(query, k=pool_k, filters=filters)

        rrf_scores: dict[str, float] = {}
        id_to_chunk: dict[str, RetrievedChunk] = {}

        for rank, chunk in enumerate(dense_results, start=1):
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
            id_to_chunk[chunk.chunk_id] = chunk

        for rank, chunk in enumerate(bm25_results, start=1):
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
            id_to_chunk[chunk.chunk_id] = chunk

        for rank, chunk in enumerate(keyword_results, start=1):
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
            id_to_chunk[chunk.chunk_id] = chunk

        rescored: list[tuple[str, float]] = []
        for cid, base_score in rrf_scores.items():
            chunk = id_to_chunk[cid]
            rescored.append((cid, base_score + self._query_metadata_boost(chunk, query, filters)))
        rescored.sort(key=lambda pair: pair[1], reverse=True)

        # When filters narrow to a single document (e.g. AAPL 10-K FY2024),
        # allow more chunks per doc so table chunks aren't capped out.
        per_doc_cap = max(1, min(3, k // 2 if k > 1 else 1))
        if filters and filters.tickers and len(filters.tickers) == 1 and filters.filing_type:
            per_doc_cap = max(per_doc_cap, k // 2)
        doc_counts: dict[str, int] = {}
        selected: list[str] = []
        overflow: list[str] = []

        for cid, _ in rescored:
            dkey = self._document_key(id_to_chunk[cid])
            if doc_counts.get(dkey, 0) < per_doc_cap:
                selected.append(cid)
                doc_counts[dkey] = doc_counts.get(dkey, 0) + 1
                if len(selected) == k:
                    break
            else:
                overflow.append(cid)

        if len(selected) < k:
            for cid in overflow:
                selected.append(cid)
                if len(selected) == k:
                    break

        rescored_map = dict(rescored)
        results = []
        for cid in selected:
            chunk = id_to_chunk[cid]
            chunk.score = rescored_map[cid]
            results.append(chunk)
        return results


def _make_chunk(chunk_id: str, text: str, score: float, meta: dict) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        score=score,
        ticker=meta.get("ticker", ""),
        company_name=meta.get("company_name", ""),
        source_type=meta.get("source_type", ""),
        filing_type=meta.get("filing_type"),
        filing_date=meta.get("filing_date"),
        year=meta.get("year"),
        quarter=meta.get("quarter"),
        period=meta.get("period"),
        period_end_date=meta.get("period_end_date"),
        fiscal_year=meta.get("fiscal_year"),
        fiscal_quarter=meta.get("fiscal_quarter"),
        canonical_period=meta.get("canonical_period"),
        fiscal_period_sort=meta.get("fiscal_period_sort"),
        section_type=meta.get("section_type"),
        chunk_kind=meta.get("chunk_kind"),
        speaker_count=meta.get("speaker_count"),
        doc_id=meta.get("doc_id"),
    )


def format_results(results: list[RetrievedChunk], preview: int = 150) -> str:
    """Human-readable summary of a result list — useful for debugging."""
    if not results:
        return "  (no results)"
    lines = []
    for i, r in enumerate(results, 1):
        if r.source_type == "sec_filing":
            source = f"{r.ticker} | {r.filing_type} | {r.filing_date}"
        else:
            source = f"{r.ticker} | ECT {r.period}"
        lines.append(f"  [{i}] score={r.score:.4f}  {source}")
        lines.append(f"       {r.text[:preview].replace(chr(10), ' ')}...")
    return "\n".join(lines)


if __name__ == "__main__":
    retriever = Retriever()

    test_cases = [
        ("revenue growth guidance",              None),
        ("capital expenditure outlook",          SearchFilters(tickers=["AAPL"], filing_type="10-K")),
        ("credit loss provisions and reserves",  SearchFilters(tickers=["JPM"], filing_type="10-K")),
        ("gross margin Q3",                      SearchFilters(source_type="ect")),
        ("risk factors supply chain disruption", SearchFilters(tickers=["AAPL", "F"], filing_type="10-K")),
    ]

    for query, filters in test_cases:
        label = repr(filters) if filters else "no filters"
        print(f"\n{'='*65}")
        print(f"Query : {query!r}")
        print(f"Filter: {label}")

        print("\nDense:")
        print(format_results(retriever.dense_search(query, k=2, filters=filters)))

        print("\nBM25:")
        print(format_results(retriever.bm25_search(query, k=2, filters=filters)))

        print("\nHybrid (RRF):")
        print(format_results(retriever.search(query, k=3, filters=filters)))
