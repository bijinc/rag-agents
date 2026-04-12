import argparse

from src.constants import CHUNK_SIZE, OVERLAP
from src.chunking import build_index
from src.baseline.evaluation import evaluate_pipeline
from src.agentic.agentic_evaluation import evaluate_agentic_pipeline


def run(latest_data=False, chunk_size=CHUNK_SIZE, overlap=OVERLAP, max_questions=None, seed=42):
    
    # Step 1: Fetch latest data if flag is set
    if latest_data:
        from src.data import fetch_data
        fetch_data()

    # Step 2: Build index (chunking, embedding, and indexing)
    build_index(chunk_size, overlap)

    # Step 3: Run Baseline RAG pipeline on questions
    evaluate_pipeline(max_questions=max_questions, seed=seed)  # runs the pipeline on the benchmark

    # Step 4: Run Agentic RAG pipeline on questions
    evaluate_agentic_pipeline(max_questions=max_questions, seed=seed)  # runs the agentic pipeline on the benchmark
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run complete Baseline vs Agentic RAG evaluation pipeline")
    parser.add_argument('-l', '--latest', action='store_true', help="Download and use the latest SEC filings and ECT transcripts (instead of existing local copies)")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Size of each text chunk for embedding")
    parser.add_argument("--overlap", type=int, default=OVERLAP, help="Number of overlapping tokens between chunks")
    parser.add_argument("--max-questions", type=int, default=None, help="Limit number of benchmark questions for faster runs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed stored in run metadata")
    args = parser.parse_args()

    run(args.latest, args.chunk_size, args.overlap, args.max_questions, args.seed)
