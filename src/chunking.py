import json
import re
import tiktoken
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer
from chonkie import SemanticChunker
from chonkie.embeddings import SentenceTransformerEmbeddings

from src.constants import EMBEDDING_MODEL, COLLECTION_NAME, CHROMA_DB_PATH
from src.finance_domain import normalize_ticker, normalize_fiscal_quarter, canonical_period_key, parse_year_quarter

SEC_CHUNK_SIZE = 864
SEC_OVERLAP = 160

ECT_CHUNK_SIZE = 480
ECT_OVERLAP = 80

BATCH_SIZE = 128    # for embedding batches
_SEMANTIC_EMBEDDINGS = None


def _safe_str(value) -> str:
    if value is None:
        return ""
    return str(value)


def _classify_section(chunk_text: str, source_type: str) -> str:
    text = chunk_text.lower()
    if source_type == "sec_filing":
        if re.search(r"item\s+1a\b|risk\s+factors", text):
            return "risk_factors"
        if re.search(r"item\s+7\b|management'?s\s+discussion|md&a", text):
            return "md&a"
        if re.search(r"item\s+8\b|financial\s+statements|balance\s+sheet|income\s+statement|cash\s+flows", text):
            return "financial_statements"
        if re.search(r"forward-looking\s+statements", text):
            return "forward_looking_statements"
        return "general"

    if re.search(r"question\s*(and|&)\s*answer|q&a", text):
        return "q_and_a"
    if re.search(r"prepared\s+remarks|opening\s+remarks|operator", text):
        return "prepared_remarks"
    return "general"


def _token_window_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    if not tokens:
        return []
    step = max(1, chunk_size - overlap)
    chunks: list[str] = []
    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i : i + chunk_size]
        if chunk_tokens:
            chunks.append(encoding.decode(chunk_tokens))
    return chunks


def _looks_like_table_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if "|" in stripped or "\t" in stripped:
        return True
    # Numeric-heavy fixed-width rows often have repeated multi-space separators.
    if re.search(r"\S\s{2,}\S", stripped) and re.search(r"\d", stripped):
        return True
    return False


def _sec_structured_segments(text: str) -> list[dict]:
    lines = text.splitlines()
    if not lines:
        return [{"kind": "prose", "text": text}]

    segments: list[dict] = []
    table_buf: list[str] = []
    prose_buf: list[str] = []

    def flush_prose() -> None:
        if prose_buf:
            prose_text = "\n".join(prose_buf).strip()
            if prose_text:
                segments.append({"kind": "prose", "text": prose_text})
            prose_buf.clear()

    def flush_table() -> None:
        if table_buf:
            table_text = "\n".join(table_buf).strip()
            if table_text:
                segments.append({"kind": "table", "text": table_text})
            table_buf.clear()

    for line in lines:
        if _looks_like_table_line(line):
            flush_prose()
            table_buf.append(line)
        else:
            flush_table()
            prose_buf.append(line)

    flush_table()
    flush_prose()

    return segments or [{"kind": "prose", "text": text}]


def _normalize_semantic_chunks(chonkie_chunks, chunk_kind: str) -> list[dict]:
    normalized_chunks: list[dict] = []
    for item in chonkie_chunks:
        text = item.text if hasattr(item, "text") else item
        if isinstance(text, str) and text.strip():
            normalized_chunks.append({"text": text, "chunk_kind": chunk_kind})
    return normalized_chunks


def _get_semantic_embeddings():
    global _SEMANTIC_EMBEDDINGS
    if _SEMANTIC_EMBEDDINGS is None:
        _SEMANTIC_EMBEDDINGS = SentenceTransformerEmbeddings("all-MiniLM-L6-v2", device="cpu")
    return _SEMANTIC_EMBEDDINGS


def _build_semantic_chunker(chunk_size: int) -> SemanticChunker:
    return SemanticChunker(embedding_model=_get_semantic_embeddings(), chunk_size=chunk_size)


def _chunk_ect(text: str) -> list[dict]:
    try:
        chunker = _build_semantic_chunker(ECT_CHUNK_SIZE)
        chunks = _normalize_semantic_chunks(chunker.chunk(text), chunk_kind="semantic")
        if chunks:
            return chunks
    except Exception as exc:  # noqa: BLE001
        print(f"  [chunking] ECT Chonkie failed, using token windows: {exc}")

    fallback = _token_window_split(text, chunk_size=ECT_CHUNK_SIZE, overlap=ECT_OVERLAP)
    return [{"text": c, "chunk_kind": "semantic"} for c in fallback if c.strip()]


def _chunk_sec(text: str) -> list[dict]:
    out: list[dict] = []
    prose_chunker = None
    prose_chunker_failed = False

    try:
        prose_chunker = _build_semantic_chunker(SEC_CHUNK_SIZE)
    except Exception as exc:
        prose_chunker_failed = True
        print(f"  [chunking] SEC prose Chonkie unavailable, using token windows: {exc}")

    for segment in _sec_structured_segments(text):
        kind = segment["kind"]
        seg_text = segment["text"]

        if kind == "table":
            table_chunks = _token_window_split(seg_text, chunk_size=min(SEC_CHUNK_SIZE, 512), overlap=0)
            out.extend({"text": t, "chunk_kind": "table"} for t in table_chunks if t.strip())
            continue

        if prose_chunker is not None and not prose_chunker_failed:
            try:
                semantic_chunks = _normalize_semantic_chunks(prose_chunker.chunk(seg_text), chunk_kind="semantic")
                if semantic_chunks:
                    out.extend(semantic_chunks)
                    continue
            except Exception as exc:
                prose_chunker_failed = True
                print(f"  [chunking] SEC prose Chonkie failed, using token windows: {exc}")

        prose_chunks = _token_window_split(seg_text, chunk_size=SEC_CHUNK_SIZE, overlap=SEC_OVERLAP)
        out.extend({"text": p, "chunk_kind": "prose"} for p in prose_chunks if p.strip())

    return out if out else [{"text": text, "chunk_kind": "prose"}]


def _chunk_document(text: str, source_type: str) -> list[dict]:
    if source_type == "ect":
        return _chunk_ect(text)
    else:
        return _chunk_sec(text)


def load_documents():
    """
    Load all documents from data/raw/sec_filings/ and data/raw/ect/

    Returns:
        list of dicts with raw content and source metadata
    """
    documents = []

    def append_json_docs(pattern, source_type, error_prefix):
        for json_file in pattern:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                doc["source_type"] = source_type
                documents.append(doc)
            except Exception as e:
                print(f"  {error_prefix} {json_file}: {e}")
                continue

    # Load SEC filings
    sec_path = Path("data/raw/sec_filings")
    if sec_path.exists():
        append_json_docs(sec_path.glob("*/*/*.json"), "sec_filing", "Error loading SEC filing")

    # Load ECT transcripts
    ect_path = Path("data/raw/ect")
    if ect_path.exists():
        append_json_docs(ect_path.glob("*/*.json"), "ect", "Error loading ECT transcript")

    return documents


def create_chunks(doc):
    """
    Create chunk dicts from a single document.
    Args:
        doc: Document dict with 'content' and metadata fields

    Returns:
        list of chunk dicts with metadata
    """
    chunk_entries = _chunk_document(doc["content"], doc["source_type"])
    chunks = []

    # Build chunk ID based on source type
    if doc.get("source_type") == "sec_filing":
        # SEC format: AAPL_10-K_20241101_0
        date = doc.get("filing_date", "unknown").replace("-", "")
        filing_type = doc.get("filing_type", "unknown")
        base_id = f"{doc['ticker']}_{filing_type}_{date}"
    else:
        # ECT format: AAPL_ect_2024_Q3_0
        period = doc.get("period", "unknown")
        base_id = f"{doc['ticker']}_ect_{period}"

    # Build period metadata once per document.
    doc_fiscal_year = _safe_str(doc.get("fiscal_year")) or None
    doc_fiscal_quarter = normalize_fiscal_quarter(_safe_str(doc.get("fiscal_quarter")) or None)
    doc_canonical_period = _safe_str(doc.get("canonical_period")) or None
    doc_period_end_date = _safe_str(doc.get("period_end_date")) or None

    # Derive from ECT period when explicit fiscal metadata is missing.
    if doc.get("source_type") == "ect" and (doc_fiscal_year is None or doc_fiscal_quarter is None):
        parsed_year, parsed_quarter = parse_year_quarter(_safe_str(doc.get("period")))
        if doc_fiscal_year is None and parsed_year is not None:
            doc_fiscal_year = str(parsed_year)
        if doc_fiscal_quarter is None and parsed_quarter is not None:
            doc_fiscal_quarter = parsed_quarter

    if doc_canonical_period is None:
        doc_canonical_period = canonical_period_key(doc_fiscal_year, doc_fiscal_quarter)

    # Create chunk dict for each text chunk
    for idx, entry in enumerate(chunk_entries):
        chunk_str = entry["text"]
        chunk_kind = entry.get("chunk_kind", "prose")
        chunk_dict = {
            "chunk_id": f"{base_id}_{idx}",
            "ticker": normalize_ticker(_safe_str(doc.get("ticker"))) or _safe_str(doc.get("ticker")),
            "source_type": doc["source_type"],
            "section_type": _classify_section(chunk_str, _safe_str(doc.get("source_type"))),
            "chunk_kind": chunk_kind,
            "period_end_date": doc_period_end_date,
            "fiscal_year": doc_fiscal_year,
            "fiscal_quarter": doc_fiscal_quarter,
            "canonical_period": doc_canonical_period,
            "text": chunk_str
        }

        # Add source-specific metadata
        if doc["source_type"] == "sec_filing":
            chunk_dict["filing_type"] = doc.get("filing_type", "")
            chunk_dict["filing_date"] = doc.get("filing_date", "")
        else:
            chunk_dict["period"] = doc.get("period", "")

        chunks.append(chunk_dict)

    return chunks


def build_index():

    print("Loading documents...")
    documents = load_documents()

    sec_count = len([d for d in documents if d.get("source_type") == "sec_filing"])
    ect_count = len([d for d in documents if d.get("source_type") == "ect"])

    print(f"  Loaded {sec_count} SEC filings")
    print(f"  Loaded {ect_count} ECT transcripts")
    print(f"  Total: {len(documents)} documents\n")

    # Delete existing collection if it exists (to allow re-runs)
    client = chromadb.PersistentClient(CHROMA_DB_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    print("Embedding and indexing chunks...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    sec_chunks = 0
    ect_chunks = 0
    total_chunks = 0
    pending_chunks: list[dict] = []
    batch_counter = 0

    def _flush_batch(batch_chunks: list[dict]) -> None:
        nonlocal batch_counter
        if not batch_chunks:
            return

        texts = [c["text"] for c in batch_chunks]
        ids = [c["chunk_id"] for c in batch_chunks]
        metadatas = [
            {
                "chunk_id": c["chunk_id"],
                "ticker": c["ticker"],
                "source_type": c["source_type"],
                "section_type": c["section_type"],
                "chunk_kind": c.get("chunk_kind", "prose"),
                "period_end_date": c.get("period_end_date"),
                "fiscal_year": c.get("fiscal_year"),
                "fiscal_quarter": c.get("fiscal_quarter"),
                "canonical_period": c.get("canonical_period"),
                **({
                    "filing_type": c["filing_type"],
                    "filing_date": c["filing_date"]
                } if c["source_type"] == "sec_filing" else {
                    "period": c["period"]
                })
            }
            for c in batch_chunks
        ]

        embeddings = model.encode(texts)
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        batch_counter += 1
        print(f"  [batch {batch_counter:4d}] Embedded {len(batch_chunks):3d} chunks")

    for doc in documents:
        chunks = create_chunks(doc)
        if doc.get("source_type") == "sec_filing":
            sec_chunks += len(chunks)
        else:
            ect_chunks += len(chunks)
        total_chunks += len(chunks)

        pending_chunks.extend(chunks)
        while len(pending_chunks) >= BATCH_SIZE:
            current_batch = pending_chunks[:BATCH_SIZE]
            pending_chunks = pending_chunks[BATCH_SIZE:]
            _flush_batch(current_batch)

    _flush_batch(pending_chunks)

    print(f"\nIndexed {total_chunks} chunks into ChromaDB")
    print(f"\nSummary:")
    print(f"  Documents: {len(documents)}")
    print(f"  Created {sec_chunks} chunks from SEC filings")
    print(f"  Created {ect_chunks} chunks from ECT transcripts")
    print(f"  Chunks: {total_chunks}")
    print(f"  Vector store: ChromaDB at {CHROMA_DB_PATH}/")
    print(f"  Collection: {COLLECTION_NAME}\n")
