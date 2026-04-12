import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from dataclasses import dataclass
from pathlib import Path

from src.constants import EMBEDDING_MODEL, COLLECTION_NAME, CHROMA_DB_PATH

##############################################################################
#                              CONFIGURATION                                 #
##############################################################################

# CHROMA_DB_PATH = str(Path(__file__).resolve().parent.parent / "data" / "chroma_db")

RRF_K = 60       # constant from the original RRF paper (Cormack et al. 2009)
DEFAULT_TOP_K = 5

##############################################################################
#                           DATA STRUCTURES                                  #
##############################################################################

@dataclass
class SearchFilters:
    """Optional filters to scope retrieval to a subset of the corpus."""
    tickers: list[str] | None = None
    source_type: str | None = None    # "sec_filing" | "ect"
    filing_type: str | None = None    # "10-K" | "10-Q" | "8-K"
    date_from: str | None = None      # "YYYY-MM-DD" — SEC filings lower bound
    date_to: str | None = None        # "YYYY-MM-DD" — SEC filings upper bound
    year: int | None = None           # ECT year
    quarter: int | None = None        # ECT quarter (1–4)


@dataclass
class RetrievedChunk:
    """A single retrieved chunk with its relevance score"""
    chunk_id: str
    text: str
    score: float
    ticker: str
    company_name: str
    source_type: str              # "sec_filing" | "ect"
    filing_type: str | None = None   # SEC only
    filing_date: str | None = None   # SEC only
    year: str | None = None          # ECT only
    quarter: str | None = None       # ECT only
    period: str | None = None        # ECT only


##############################################################################
#                              RETRIEVER                                     #
##############################################################################

class Retriever:
    """
    Hybrid retriever combining:
      - Dense search  : ChromaDB cosine similarity on all-MiniLM-L6-v2 embeddings
      - Sparse search : BM25Okapi over an in-memory copy of the corpus
      - Fusion        : Reciprocal Rank Fusion (RRF) as the primary entry point

    The BM25 corpus is loaded once at init from ChromaDB so both paths
    operate on exactly the same set of chunks.
    """

    def __init__(self):
        print("Initializing Retriever...")

        self._client = chromadb.PersistentClient(CHROMA_DB_PATH)
        self._collection = self._client.get_collection(COLLECTION_NAME)
        print(f"ChromaDB: '{COLLECTION_NAME}' ({self._collection.count()} chunks)")

        self._embed_model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"Embedding model: {EMBEDDING_MODEL}")

        self._load_bm25_corpus()
        print(f"BM25 corpus: {len(self._corpus_ids)} chunks\n")

    def _load_bm25_corpus(self):
        """Pull every chunk out of ChromaDB and pre-tokenize for BM25."""
        result = self._collection.get(include=["documents", "metadatas"])
        self._corpus_ids: list[str] = result["ids"]
        self._corpus_texts: list[str] = result["documents"]
        self._corpus_metadatas: list[dict] = result["metadatas"]
        # Simple whitespace tokenisation is sufficient for BM25
        self._tokenized_corpus = [text.lower().split() for text in self._corpus_texts]

    # -------------------------------------------------------------------------
    # Filter helpers
    # -------------------------------------------------------------------------

    def _build_chroma_filter(self, filters: SearchFilters | None) -> dict | None:
        """
        Translate SearchFilters into a ChromaDB 'where' clause.
        Returns None when no filter is needed so the caller can skip the kwarg.
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

        if filters.filing_type:
            clauses.append({"filing_type": {"$eq": filters.filing_type}})

        if filters.date_from:
            clauses.append({"filing_date": {"$gte": filters.date_from}})

        if filters.date_to:
            clauses.append({"filing_date": {"$lte": filters.date_to}})

        if filters.year is not None:
            clauses.append({"year": {"$eq": str(filters.year)}})

        if filters.quarter is not None:
            clauses.append({"quarter": {"$eq": str(filters.quarter)}})

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
        if filters.filing_type and meta.get("filing_type") != filters.filing_type:
            return False
        if filters.date_from and meta.get("filing_date", "") < filters.date_from:
            return False
        if filters.date_to and meta.get("filing_date", "") > filters.date_to:
            return False
        if filters.year is not None and meta.get("year") != str(filters.year):
            return False
        if filters.quarter is not None and meta.get("quarter") != str(filters.quarter):
            return False
        return True

    # -------------------------------------------------------------------------
    # Dense retrieval
    # -------------------------------------------------------------------------

    def dense_search(self,query: str,k: int = DEFAULT_TOP_K,filters: SearchFilters | None = None,) -> list[RetrievedChunk]:
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

        return [
            _make_chunk(cid, doc, 1.0 / (1.0 + dist), meta)
            for cid, doc, dist, meta in zip(
                result["ids"][0],
                result["documents"][0],
                result["distances"][0],
                result["metadatas"][0],
            )
        ]

    # -------------------------------------------------------------------------
    # BM25 retrieval
    # -------------------------------------------------------------------------

    def bm25_search(self,query: str,k: int = DEFAULT_TOP_K,filters: SearchFilters | None = None,) -> list[RetrievedChunk]:
        """
        Top-k retrieval by BM25Okapi keyword scoring.
        The corpus is filtered in-memory before scoring so that BM25 sees
        the same slice of documents as the dense search.
        """
        indices = [
            i for i, meta in enumerate(self._corpus_metadatas)
            if self._metadata_matches(meta, filters)
        ]
        if not indices:
            return []

        bm25 = BM25Okapi([self._tokenized_corpus[i] for i in indices])
        scores = bm25.get_scores(query.lower().split())

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

    # -------------------------------------------------------------------------
    # Hybrid retrieval — primary entry point
    # -------------------------------------------------------------------------

    def search(self,query: str, k: int = DEFAULT_TOP_K,filters: SearchFilters | None = None,) -> list[RetrievedChunk]:
        """
        Hybrid search via Reciprocal Rank Fusion.

        Each system contributes 2*k candidates; a chunk that ranks highly in
        both lists receives the highest fused score.  RRF score for document d:

            score(d) = Σ  1 / (RRF_K + rank_i(d))
                      lists

        RRF_K=60 is the standard value from Cormack et al. (2009).
        """
        dense_results = self.dense_search(query, k=k * 2, filters=filters)
        bm25_results = self.bm25_search(query, k=k * 2, filters=filters)

        rrf_scores: dict[str, float] = {}
        id_to_chunk: dict[str, RetrievedChunk] = {}

        for rank, chunk in enumerate(dense_results, start=1):
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
            id_to_chunk[chunk.chunk_id] = chunk

        for rank, chunk in enumerate(bm25_results, start=1):
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
            id_to_chunk[chunk.chunk_id] = chunk

        top_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:k]

        results = []
        for cid in top_ids:
            chunk = id_to_chunk[cid]
            chunk.score = rrf_scores[cid]
            results.append(chunk)

        return results


##############################################################################
#                              HELPERS                                       #
##############################################################################

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


##############################################################################
#                              SMOKE TEST                                    #
##############################################################################

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
