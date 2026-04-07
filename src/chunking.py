import os
import json
import tiktoken
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

##############################################################################
#                              CONFIGURATION                                 #
##############################################################################

CHUNK_SIZE = 512  # tokens per chunk
OVERLAP = 50      # overlap tokens between chunks
BATCH_SIZE = 64   # for embedding batches
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_DB_PATH = "data/chroma_db"
COLLECTION_NAME = "financial_docs"

##############################################################################
#                          DOCUMENT LOADING                                  #
##############################################################################

def load_documents():
    """
    Load all documents from data/raw/sec_filings/ and data/raw/ect/

    Returns:
        list of dicts with keys: ticker, company_name, content, source_type, ...
    """
    documents = []

    # Load SEC filings
    sec_path = Path("data/raw/sec_filings")
    if sec_path.exists():
        for ticker_dir in sec_path.iterdir():
            if not ticker_dir.is_dir():
                continue
            ticker = ticker_dir.name

            for filing_type_dir in ticker_dir.iterdir():
                if not filing_type_dir.is_dir():
                    continue
                filing_type = filing_type_dir.name

                for json_file in filing_type_dir.glob("*.json"):
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            doc = json.load(f)
                        doc["source_type"] = "sec_filing"
                        documents.append(doc)
                    except Exception as e:
                        print(f"  ✗ Error loading {json_file}: {e}")
                        continue

    # Load ECT transcripts
    ect_path = Path("data/raw/ect")
    if ect_path.exists():
        for ticker_dir in ect_path.iterdir():
            if not ticker_dir.is_dir():
                continue
            ticker = ticker_dir.name

            for json_file in ticker_dir.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        doc = json.load(f)
                    doc["source_type"] = "ect"
                    documents.append(doc)
                except Exception as e:
                    print(f"  ✗ Error loading {json_file}: {e}")
                    continue

    return documents

##############################################################################
#                              CHUNKING                                      #
##############################################################################

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """
    Split text into token-based chunks with overlap.

    Args:
        text: Input text string
        chunk_size: Target chunk size in tokens
        overlap: Overlap in tokens between consecutive chunks

    Returns:
        list of chunk text strings
    """
    # Initialize tokenizer
    encoding = tiktoken.get_encoding("cl100k_base")

    # Encode text to tokens
    tokens = encoding.encode(text)

    # Create chunks with overlap
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk_tokens = tokens[i : i + chunk_size]
        if len(chunk_tokens) > 0:
            chunk_text = encoding.decode(chunk_tokens)
            chunks.append(chunk_text)

    return chunks

def create_chunks(doc):
    """
    Create chunk dicts from a single document.

    Args:
        doc: Document dict with 'content' and metadata fields

    Returns:
        list of chunk dicts with metadata
    """
    # Get text chunks
    text_chunks = chunk_text(doc["content"], CHUNK_SIZE, OVERLAP)
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

    # Create chunk dict for each text chunk
    for idx, chunk_str in enumerate(text_chunks):
        chunk_dict = {
            "chunk_id": f"{base_id}_{idx}",
            "ticker": doc["ticker"],
            "company_name": doc.get("company_name", "Unknown"),
            "source_type": doc["source_type"],
            "chunk_index": str(idx),
            "total_chunks": str(len(text_chunks)),
            "text": chunk_str
        }

        # Add source-specific metadata
        if doc["source_type"] == "sec_filing":
            chunk_dict["filing_type"] = doc.get("filing_type", "")
            chunk_dict["filing_date"] = doc.get("filing_date", "")
        else:
            chunk_dict["year"] = str(doc.get("year", ""))
            chunk_dict["quarter"] = str(doc.get("quarter", ""))
            chunk_dict["period"] = doc.get("period", "")

        chunks.append(chunk_dict)

    return chunks

##############################################################################
#                         EMBEDDING & INDEXING                               #
##############################################################################

def build_index(chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """
    Main orchestrator: load documents, chunk, embed, and index in ChromaDB.
    """
    print("\n" + "="*70)
    print("CHUNKING AND EMBEDDING PHASE")
    print("="*70 + "\n")

    # Step 1: Load documents
    print("Loading documents...")
    documents = load_documents()

    sec_count = len([d for d in documents if d.get("source_type") == "sec_filing"])
    ect_count = len([d for d in documents if d.get("source_type") == "ect"])

    print(f"  ✓ Loaded {sec_count} SEC filings")
    print(f"  ✓ Loaded {ect_count} ECT transcripts")
    print(f"  Total: {len(documents)} documents\n")

    # Step 2: Create chunks
    print("Chunking documents...")
    all_chunks = []
    for doc in documents:
        chunks = create_chunks(doc)
        all_chunks.extend(chunks)

    sec_chunks = len([c for c in all_chunks if c.get("source_type") == "sec_filing"])
    ect_chunks = len([c for c in all_chunks if c.get("source_type") == "ect"])

    print(f"  ✓ Created {sec_chunks} chunks from SEC filings")
    print(f"  ✓ Created {ect_chunks} chunks from ECT transcripts")
    print(f"  Total: {len(all_chunks)} chunks\n")

    # Step 3: Initialize ChromaDB
    print("Initializing ChromaDB...")
    client = chromadb.PersistentClient(CHROMA_DB_PATH)

    # Delete existing collection if it exists (to allow re-runs)
    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass

    collection = client.create_collection(COLLECTION_NAME)
    print(f"  ✓ Created ChromaDB collection: {COLLECTION_NAME}\n")

    # Step 4: Initialize embedding model
    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"  ✓ Model loaded (384 dimensions)\n")

    # Step 5: Embed and upsert in batches
    print("Embedding and indexing chunks...")
    total_batches = (len(all_chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_num in range(0, len(all_chunks), BATCH_SIZE):
        batch_chunks = all_chunks[batch_num : batch_num + BATCH_SIZE]

        # Extract texts and IDs
        texts = [c["text"] for c in batch_chunks]
        ids = [c["chunk_id"] for c in batch_chunks]

        # Prepare metadata (all fields must be strings for ChromaDB)
        metadatas = [
            {
                "ticker": c["ticker"],
                "company_name": c["company_name"],
                "source_type": c["source_type"],
                "chunk_index": c["chunk_index"],
                "total_chunks": c["total_chunks"],
                **({
                    "filing_type": c["filing_type"],
                    "filing_date": c["filing_date"]
                } if c["source_type"] == "sec_filing" else {
                    "year": c["year"],
                    "quarter": c["quarter"],
                    "period": c["period"]
                })
            }
            for c in batch_chunks
        ]

        # Embed texts
        embeddings = model.encode(texts)

        # Upsert to ChromaDB
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

        batch_num_display = (batch_num // BATCH_SIZE) + 1
        print(f"  [{batch_num_display:3d}/{total_batches:3d}] Embedded {len(batch_chunks):3d} chunks")

    print(f"\n✓ Indexed {len(all_chunks)} chunks into ChromaDB")
    print(f"✓ Collection: {COLLECTION_NAME}")
    print(f"✓ Persistent storage: {CHROMA_DB_PATH}/\n")

    # Step 6: Print summary
    print("="*70)
    print("INDEXING COMPLETE")
    print("="*70)
    print(f"\nSummary:")
    print(f"  Documents: {len(documents)}")
    print(f"  Chunks: {len(all_chunks)}")
    print(f"  Embedding model: {EMBEDDING_MODEL} (384 dimensions)")
    print(f"  Vector store: ChromaDB at {CHROMA_DB_PATH}/")
    print(f"  Collection: {COLLECTION_NAME}")
    print()

##############################################################################
#                              MAIN                                          #
##############################################################################

if __name__ == "__main__":
    build_index(CHUNK_SIZE, OVERLAP)
