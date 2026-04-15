import argparse
from src.baseline.evaluation import evaluate_pipeline
from src.agentic.agentic_evaluation import evaluate_agentic_pipeline
from src.evaluation_summary import run_summary
from src.constants import (
    BASELINE_RESULTS_PATH,
    AGENTIC_RESULTS_PATH,
    SUMMARY_CHARTS_DIR,
    SUMMARY_CHARTS_PREFIX,
)


def _print_pipeline_overview(label: str, summary) -> None:
    overall = summary.aggregate_overall()
    if not overall:
        print(f"\n{label} overview: no records")
        return

    row = overall[0]
    ragas = f"{row['ragas_mean']:.3f}" if row["ragas_mean"] is not None else "n/a"
    judge = f"{row['judge_mean']:.3f}" if row["judge_mean"] is not None else "n/a"
    halluc = f"{row['hallucination_rate']:.3f}" if row["hallucination_rate"] is not None else "n/a"
    print(
        f"\n{label} overview: n={row['n_total']} | "
        f"ragas_mean={ragas} | judge_mean={judge} | halluc_rate={halluc}"
    )


def run(latest_data=False, max_questions=None, seed=42, reindex=False):
    
    # Step 1: Fetch latest data if flag is set
    if latest_data:
        from src.data import fetch_data
        fetch_data()

    # Step 2: Build index (chunking, embedding, and indexing)
    if reindex:
        from src.chunking import build_index
        build_index()

    # Step 3: Run Baseline RAG pipeline on questions
    _, baseline_summary = evaluate_pipeline(max_questions=max_questions, seed=seed)
    _print_pipeline_overview("Baseline", baseline_summary)

    # Step 4: Run Agentic RAG pipeline on questions
    _, agentic_summary = evaluate_agentic_pipeline(max_questions=max_questions, seed=seed)
    _print_pipeline_overview("Agentic", agentic_summary)

    # Step 5: Build comparative summary and charts
    _, chart_paths = run_summary(
        baseline_path=BASELINE_RESULTS_PATH,
        agentic_path=AGENTIC_RESULTS_PATH,
        output_dir=SUMMARY_CHARTS_DIR,
        prefix=SUMMARY_CHARTS_PREFIX,
    )

    if chart_paths:
        print("\nEvaluation summary charts:")
        for path in chart_paths:
            print(f"  - {path}")
    else:
        print("\nEvaluation summary skipped: no records found to visualize.")
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run complete Baseline vs Agentic RAG evaluation pipeline")
    parser.add_argument('-l', '--latest', action='store_true', help="Download and use the latest SEC filings and ECT transcripts (instead of existing local copies)")
    parser.add_argument('--reindex', action='store_true', help="Rebuild the Chroma collection from scratch")
    parser.add_argument("--max-questions", type=int, default=None, help="Limit number of benchmark questions for faster runs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed stored in run metadata")
    args = parser.parse_args()

    run(args.latest, args.max_questions, args.seed, args.reindex)
