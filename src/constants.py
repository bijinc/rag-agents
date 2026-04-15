
# EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "financial_docs"
CHROMA_DB_PATH = "data/chroma_db"


DEFAULT_TOP_K = 15
DEFAULT_SEED = 42


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GENERATOR_MODEL = "qwen/qwen-2.5-72b-instruct"
EVALUATOR_MODEL = "meta-llama/llama-3.1-8b-instruct"
DEFAULT_MODEL = "qwen"
MODELS = {
    "qwen":  GENERATOR_MODEL,
    "llama": EVALUATOR_MODEL,
}


BENCHMARK_PATH = "qa_benchmark.json"

RESULTS_DIR = "results"
BASELINE_RESULTS_PATH = "results/baseline_eval_results.json"
AGENTIC_RESULTS_PATH = "results/agentic_eval_results.json"
SUMMARY_CHARTS_DIR = "results/eval_summary_charts"
SUMMARY_CHARTS_PREFIX = "comparison"