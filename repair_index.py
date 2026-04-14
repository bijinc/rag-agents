"""repair_index.py

Rebuilds the ChromaDB HNSW vector index from existing SQLite data.
The current index is corrupted (only 100 of 10,843 vectors indexed).

Steps:
  1. Export all documents + metadata from the existing collection
  2. Delete the corrupted collection
  3. Re-create it and batch-add all chunks (triggers fresh embedding + HNSW build)

Runtime: ~5-10 minutes (re-embeds 10K chunks with all-MiniLM-L6-v2)
"""

import time
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

CHROMA_DB_PATH = "data/chroma_db"
COLLECTION_NAME = "financial_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EXPORT_BATCH = 500      # how many to read per batch from old collection
INSERT_BATCH = 256      # how many to insert per batch (embedding bottleneck)


def main():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    old_col = client.get_collection(COLLECTION_NAME)
    total = old_col.count()
    print(f"Existing collection: {total} chunks")

    # ── Step 1: Export all docs + metadata ──────────────────────────
    print(f"\n[1/3] Exporting {total} chunks from corrupted collection...")
    all_ids = []
    all_docs = []
    all_metas = []

    offset = 0
    while offset < total:
        batch = old_col.get(
            limit=EXPORT_BATCH,
            offset=offset,
            include=["metadatas", "documents"],
        )
        all_ids.extend(batch["ids"])
        all_docs.extend(batch["documents"])
        all_metas.extend(batch["metadatas"])
        offset += len(batch["ids"])
        print(f"  exported {offset}/{total}")

    print(f"  Total exported: {len(all_ids)} chunks")

    # Sanity check for duplicates
    if len(set(all_ids)) != len(all_ids):
        dupes = len(all_ids) - len(set(all_ids))
        print(f"  WARNING: {dupes} duplicate IDs found, deduplicating...")
        seen = set()
        deduped = {"ids": [], "docs": [], "metas": []}
        for i, cid in enumerate(all_ids):
            if cid not in seen:
                seen.add(cid)
                deduped["ids"].append(cid)
                deduped["docs"].append(all_docs[i])
                deduped["metas"].append(all_metas[i])
        all_ids = deduped["ids"]
        all_docs = deduped["docs"]
        all_metas = deduped["metas"]
        print(f"  After dedup: {len(all_ids)} chunks")

    # ── Step 2: Delete old collection ───────────────────────────────
    print(f"\n[2/3] Deleting corrupted collection...")
    client.delete_collection(COLLECTION_NAME)
    print("  Deleted.")

    # ── Step 3: Re-create and batch insert ──────────────────────────
    print(f"\n[3/3] Re-creating collection and inserting {len(all_ids)} chunks...")
    embed_fn = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
    )
    new_col = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    t0 = time.time()
    for start in range(0, len(all_ids), INSERT_BATCH):
        end = min(start + INSERT_BATCH, len(all_ids))
        new_col.add(
            ids=all_ids[start:end],
            documents=all_docs[start:end],
            metadatas=all_metas[start:end],
        )
        elapsed = time.time() - t0
        rate = (end) / elapsed if elapsed > 0 else 0
        eta = (len(all_ids) - end) / rate if rate > 0 else 0
        print(f"  inserted {end}/{len(all_ids)}  ({rate:.0f} chunks/s, ETA {eta:.0f}s)")

    # ── Verify ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"DONE in {time.time() - t0:.1f}s")
    print(f"Collection size: {new_col.count()}")

    # Quick retrieval test
    test = new_col.query(
        query_texts=["Apple total net revenue fiscal year 2024"],
        n_results=3,
        where={"ticker": "AAPL"},
    )
    print(f"\nVerification query (AAPL revenue):")
    for i, (doc, meta, dist) in enumerate(
        zip(test["documents"][0], test["metadatas"][0], test["distances"][0])
    ):
        print(f"  [{i+1}] dist={dist:.4f} {meta.get('ticker')} {meta.get('filing_type')} {meta.get('filing_date','')}")
        print(f"      {doc[:200]}")
    print()


if __name__ == "__main__":
    main()
