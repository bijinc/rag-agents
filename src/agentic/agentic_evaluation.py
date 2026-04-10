"""src/agentic_evaluation.py

RQ2 evaluation: Does agentic RAG reduce hallucination vs single-pass baseline?

Runs the AgenticPipeline on the same QA benchmark used by evaluation.py,
then applies the same RAGAS faithfulness and LLM-as-judge metrics so results
are directly comparable.

Extra fields vs evaluation.py's EvalRecord:
  - sub_questions        : decomposed sub-questions from query_analyzer
  - retrieval_attempts   : how many retrieval retries were made
  - gate_faithful        : faithfulness verdict from the pipeline's own gate
  - gate_confidence      : "high" | "low" from the pipeline's faithfulness gate
  - question_type        : "L1" | "L2" | "L3" | "L4" from query_analyzer

Usage:
    python -m src.agentic_evaluation
    python -m src.agentic_evaluation --skip-ragas
    python -m src.agentic_evaluation --skip-judge
    python -m src.agentic_evaluation --judge-only
"""

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.agentic.agentic_pipeline import AgenticPipeline, AgenticResult
from src.baseline.evaluation import run_ragas, run_llm_judge, EvalRecord

##############################################################################
#                              CONFIGURATION                                 #
##############################################################################

GENERATOR_MODEL = "qwen/qwen-2.5-7b-instruct"
EVALUATOR_MODEL = "meta-llama/llama-3.1-8b-instruct"

BENCHMARK_PATH = "data/qa_benchmark.json"
RESULTS_PATH   = "data/agentic_eval_results.json"

##############################################################################
#                           DATA STRUCTURES                                  #
##############################################################################

@dataclass
class AgenticEvalRecord(EvalRecord):
    """EvalRecord extended with agentic-pipeline-specific metadata."""
    sub_questions:      list[str] = None
    retrieval_attempts: int       = 0
    gate_faithful:      bool      = True
    gate_confidence:    str       = "high"
    question_type:      str       = "L1"

    def __post_init__(self):
        if self.sub_questions is None:
            self.sub_questions = []


##############################################################################
#                        STEP 1 — RUN AGENTIC PIPELINE                      #
##############################################################################

def run_agentic_pipeline(
    benchmark_path: str = BENCHMARK_PATH,
) -> list[AgenticEvalRecord]:
    """Run the AgenticPipeline on every benchmark question."""
    with open(benchmark_path) as f:
        benchmark = json.load(f)

    pipeline = AgenticPipeline(model="qwen")
    records: list[AgenticEvalRecord] = []

    print(f"\nRunning agentic pipeline on {len(benchmark)} questions  [generator: {GENERATOR_MODEL}]")
    for i, item in enumerate(benchmark, 1):
        print(f"  [{i:2d}/{len(benchmark)}] {item['question'][:70]}")
        result: AgenticResult = pipeline.run(item["question"])
        records.append(AgenticEvalRecord(
            id=item["id"],
            question=item["question"],
            difficulty=item["difficulty"],
            tickers=item.get("tickers", []),
            ground_truth=item["answer"],
            generated_answer=result.answer,
            contexts=result.contexts,
            source_refs=item.get("source_refs", []),
            generator_model=GENERATOR_MODEL,
            evaluator_model=EVALUATOR_MODEL,
            top_k=5,
            # agentic-specific
            sub_questions=result.sub_questions,
            retrieval_attempts=result.retrieval_attempts,
            gate_faithful=result.is_faithful,
            gate_confidence=result.confidence,
            question_type=result.question_type,
        ))
        time.sleep(0.3)   # avoid rate-limiting between questions

    return records


##############################################################################
#                          SAVE + SUMMARY                                    #
##############################################################################

def save_results(records: list[AgenticEvalRecord], path: str = RESULTS_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([asdict(r) for r in records], f, indent=2)
    print(f"\nResults saved → {path}")


def print_summary(records: list[AgenticEvalRecord]) -> None:
    print(f"\n{'='*65}")
    print(f"AGENTIC EVALUATION SUMMARY  ({len(records)} questions)")
    print(f"  Generator : {GENERATOR_MODEL}")
    print(f"  Evaluator : {EVALUATOR_MODEL}")
    print(f"{'='*65}")

    ragas_scores = [r.ragas_faithfulness for r in records if r.ragas_faithfulness is not None]
    judge_scores = [r.judge_score        for r in records if r.judge_score        is not None]
    hallucinated = sum(1 for r in records if r.judge_hallucination)
    retried      = sum(1 for r in records if r.retrieval_attempts > 0)
    gate_fails   = sum(1 for r in records if not r.gate_faithful)

    def fmt(scores: list[float]) -> str:
        if not scores:
            return "n/a"
        return f"mean={sum(scores)/len(scores):.3f}  min={min(scores):.3f}  max={max(scores):.3f}"

    print(f"RAGAS Faithfulness    : {fmt(ragas_scores)}")
    print(f"LLM Judge Score       : {fmt(judge_scores)}")
    print(f"Hallucinations (judge): {hallucinated}/{len(records)}")
    print(f"Gate flagged unfaithful: {gate_fails}/{len(records)}")
    print(f"Questions with retries : {retried}/{len(records)}")

    # Breakdown by difficulty
    print(f"\n{'─'*75}")
    print(f"{'Level':<8} {'N':>4}  {'RAGAS':>7}  {'Judge':>7}  {'Halluc':>8}  {'Retries':>9}  {'Gate❌':>7}")
    print(f"{'─'*75}")
    for diff in ["L1", "L2", "L3", "L4"]:
        sub = [r for r in records if r.difficulty == diff]
        if not sub:
            continue
        rs = [r.ragas_faithfulness for r in sub if r.ragas_faithfulness is not None]
        js = [r.judge_score        for r in sub if r.judge_score        is not None]
        hc = sum(1 for r in sub if r.judge_hallucination)
        rt = sum(r.retrieval_attempts for r in sub)
        gf = sum(1 for r in sub if not r.gate_faithful)
        r_mean = f"{sum(rs)/len(rs):.3f}" if rs else "  —"
        j_mean = f"{sum(js)/len(js):.3f}" if js else "  —"
        print(f"{diff:<8} {len(sub):>4}  {r_mean:>7}  {j_mean:>7}  {hc:>4}/{len(sub)}  {rt:>6} ret  {gf:>3}/{len(sub)}")
    print(f"{'='*75}")


##############################################################################
#                          RELOAD SAVED RECORDS                              #
##############################################################################

def load_records(path: str) -> list[AgenticEvalRecord]:
    with open(path) as f:
        return [AgenticEvalRecord(**item) for item in json.load(f)]


##############################################################################
#                                ENTRY POINT                                 #
##############################################################################

def evaluate_agentic_pipeline(
    benchmark_path: str = BENCHMARK_PATH,
    results_path: str   = RESULTS_PATH,
    skip_ragas: bool    = False,
    skip_judge: bool    = False,
    judge_only: bool    = False,
) -> list[AgenticEvalRecord]:

    if judge_only:
        print(f"Loading existing results from {results_path}")
        records = load_records(results_path)
    else:
        records = run_agentic_pipeline(benchmark_path)
        if not skip_ragas:
            run_ragas(records)          # reused from evaluation.py

    if not skip_judge:
        run_llm_judge(records)          # reused from evaluation.py

    save_results(records, results_path)
    print_summary(records)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run agentic RAG evaluation")
    parser.add_argument("--skip-ragas",  action="store_true", help="Skip RAGAS faithfulness step")
    parser.add_argument("--skip-judge",  action="store_true", help="Skip LLM-as-judge step")
    parser.add_argument("--judge-only",  action="store_true", help="Load existing results, run judge only")
    parser.add_argument("--benchmark",   default=BENCHMARK_PATH)
    parser.add_argument("--output",      default=RESULTS_PATH)
    args = parser.parse_args()

    evaluate_agentic_pipeline(
        benchmark_path=args.benchmark,
        results_path=args.output,
        skip_ragas=args.skip_ragas,
        skip_judge=args.skip_judge,
        judge_only=args.judge_only,
    )
