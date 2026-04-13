
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "financial_docs"
CHROMA_DB_PATH = "data/chroma_db"


DEFAULT_TOP_K = 10
DEFAULT_SEED = 42
SEC_CHUNKER_BACKEND = "token"  # "token" | "chonkie"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GENERATOR_MODEL = "qwen/qwen-2.5-7b-instruct"
EVALUATOR_MODEL = "meta-llama/llama-3.1-8b-instruct"
DEFAULT_MODEL = "qwen"
MODELS = {
    "qwen":  GENERATOR_MODEL,
    "llama": EVALUATOR_MODEL,
}