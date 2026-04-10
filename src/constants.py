
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "financial_docs"
CHROMA_DB_PATH = "data/chroma_db"
# CHROMA_DB_PATH = str(Path(__file__).resolve().parent.parent / "data" / "chroma_db")


CHUNK_SIZE = 512  # tokens per chunk
OVERLAP = 50      # overlap tokens between chunks
BATCH_SIZE = 64   # for embedding batches