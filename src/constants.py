
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "financial_docs"
CHROMA_DB_PATH = "data/chroma_db"
# CHROMA_DB_PATH = str(Path(__file__).resolve().parent.parent / "data" / "chroma_db")


CHUNK_SIZE = 512  # tokens per chunk
OVERLAP = 50      # overlap tokens between chunks
BATCH_SIZE = 64   # for embedding batches


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GENERATOR_MODEL = "qwen/qwen-2.5-7b-instruct"
EVALUATOR_MODEL = "meta-llama/llama-3.1-8b-instruct"
DEFAULT_MODEL = "qwen"
MODELS = {
    "qwen":  GENERATOR_MODEL,
    "llama": EVALUATOR_MODEL,
}