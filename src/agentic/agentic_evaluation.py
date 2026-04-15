"""
Runs the AgenticPipeline on the same QA benchmark used by evaluation.py,
then applies the same RAGAS faithfulness and LLM-as-judge metrics so results
are directly comparable.

Usage:
    python -m src.agentic_evaluation
    python -m src.agentic_evaluation --skip-ragas
    python -m src.agentic_evaluation --skip-judge
    python -m src.agentic_evaluation --judge-only
"""

import argparse
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4
from dotenv import load_dotenv

from src.constants import GENERATOR_MODEL, EVALUATOR_MODEL
from src.agentic.agentic_pipeline import DEFAULT_MODEL, AgenticPipeline, AgenticResult
from src.baseline.evaluation import run_ragas, run_llm_judge, EvalRecord
from src.eval_utils import (
    load_json,
    save_json,
    utc_now_iso,
    validate_benchmark_items,
    validate_eval_flags,
)

##############################################################################
#                              CONFIGURATION                                 #
##############################################################################

load_dotenv()

BENCHMARK_PATH = "data/qa_benchmark.json"
RESULTS_PATH   = "data/agentic_eval_results.json"
DEFAULT_SEED   = 42

##############################################################################
#                           DATA STRUCTURES                                  #
##############################################################################

@dataclass
class AgenticEvalRecord(EvalRecord):
    """EvalRecord extended with agentic-pipeline-specific metadata."""
    sub_questions:      list[str] = None
    retrieval_attempts: int       = 0
    gate_faithful:      bool | None = True
    gate_status:        str       = "unknown"
    gate_confidence:    str       = "high"
    question_type:      str       = "L1"
    retrieval_source_recall: float = 0.0
    retrieval_source_hits:   int   = 0
    retrieval_source_total:  int   = 0
    retrieved_source_refs:   list[str] | None = None
    citation_count:      int = 0
    node_errors:        dict[str, str] | None = None
    pipeline_succeeded: bool = True
    diagnostics:        dict | None = None

    def __post_init__(self):
        if self.sub_questions is None:
            self.sub_questions = []
        if self.node_errors is None:
            self.node_errors = {}
        if self.diagnostics is None:
            self.diagnostics = {}
        if self.retrieved_source_refs is None:
            self.retrieved_source_refs = []


def _sec_ref_from_metadata(ticker: str, filing_type: str | None, filing_date: str | None) -> str | None:
    if not ticker or not filing_type or not filing_date:
        return None
    compact_date = filing_date.replace("-", "")
    return f"sec_filings/{ticker}/{filing_type}/{filing_type}_{compact_date}.json"


def _ect_ref_from_metadata(ticker: str, period: str | None) -> str | None:
    if not ticker or not period:
        return None
    return f"ect/{ticker}/{period}.json"


def _normalize_ref(ref: str) -> str:
    return re.sub(r"\\+", "/", (ref or "").strip())


def _estimate_retrieved_source_refs(result: AgenticResult) -> set[str]:
    refs: set[str] = set()
    for chunk in result.retrieved_chunks:
        if chunk.source_type == "sec_filing":
            ref = _sec_ref_from_metadata(chunk.ticker, chunk.filing_type, chunk.filing_date)
        else:
            ref = _ect_ref_from_metadata(chunk.ticker, chunk.period)
        if ref:
            refs.add(_normalize_ref(ref))
    return refs


def _compute_source_recall(expected_refs: list[str], retrieved_refs: set[str]) -> tuple[float, int, int]:
    expected = {_normalize_ref(r) for r in expected_refs or [] if r}
    if not expected:
        return (1.0, 0, 0)
    hits = len(expected & retrieved_refs)
    total = len(expected)
    return (hits / total, hits, total)


##############################################################################
#                        STEP 1 — RUN AGENTIC PIPELINE                      #
##############################################################################

def run_agentic_pipeline(
    benchmark: list[dict],
    run_metadata: dict | None = None,
) -> list[AgenticEvalRecord]:
    """Run the AgenticPipeline on every benchmark question."""
    run_metadata = run_metadata or {}

    pipeline = AgenticPipeline(model=DEFAULT_MODEL)
    records: list[AgenticEvalRecord] = []

    print(f"\nRunning agentic pipeline on {len(benchmark)} questions  [generator: {GENERATOR_MODEL}]")
    for i, item in enumerate(benchmark, 1):
        print(f"  [{i:2d}/{len(benchmark)}] {item['question'][:70]}")
        result: AgenticResult = pipeline.run(item["question"])
        retrieved_refs = _estimate_retrieved_source_refs(result)
        recall, hits, total = _compute_source_recall(item.get("source_refs", []), retrieved_refs)
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
            gate_status=result.faithfulness_status,
            gate_confidence=result.confidence,
            question_type=result.question_type,
            retrieval_source_recall=round(float(recall), 4),
            retrieval_source_hits=hits,
            retrieval_source_total=total,
            retrieved_source_refs=sorted(retrieved_refs),
            citation_count=len(result.citations),
            node_errors=result.node_errors,
            pipeline_succeeded=not bool(result.node_errors),
            diagnostics=result.diagnostics,
            run_id=run_metadata.get("run_id", ""),
            run_started_at=run_metadata.get("run_started_at", ""),
            run_seed=run_metadata.get("seed", DEFAULT_SEED),
        ))
        time.sleep(0.3)   # avoid rate-limiting between questions

    return records


##############################################################################
#                          SAVE + SUMMARY                                    #
##############################################################################

def save_results(records: list[AgenticEvalRecord], path: str = RESULTS_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_json(path, [asdict(r) for r in records])
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
    gate_unknown = sum(1 for r in records if r.gate_faithful is None)
    gate_passed  = sum(1 for r in records if r.gate_status == "passed")
    gate_failed  = sum(1 for r in records if r.gate_status == "failed")
    pipeline_failed = sum(1 for r in records if not r.pipeline_succeeded)
    source_recalls = [r.retrieval_source_recall for r in records]
    avg_source_recall = sum(source_recalls) / len(source_recalls) if source_recalls else 0.0
    any_source_hit = sum(1 for r in records if r.retrieval_source_hits > 0)

    def fmt(scores: list[float]) -> str:
        if not scores:
            return "n/a"
        return f"mean={sum(scores)/len(scores):.3f}  min={min(scores):.3f}  max={max(scores):.3f}"

    print(f"RAGAS Faithfulness     : {fmt(ragas_scores)}")
    print(f"LLM Judge Score        : {fmt(judge_scores)}")
    print(f"Hallucinations (judge) : {hallucinated}/{len(records)}")
    print(f"Gate flagged unfaithful: {gate_fails}/{len(records)}")
    print(f"Gate unknown           : {gate_unknown}/{len(records)}")
    print(f"Gate status counts     : passed={gate_passed}, failed={gate_failed}, unknown={gate_unknown}")
    print(f"Questions with retries : {retried}/{len(records)}")
    print(f"Pipeline error records : {pipeline_failed}/{len(records)}")
    print(f"Source recall (avg)    : {avg_source_recall:.3f}")
    print(f"Any expected source hit: {any_source_hit}/{len(records)}")

    # Retry diagnostics from sufficiency checks
    llm_sufficiency_calls = 0
    heuristic_blocks = 0
    retry_attempt_counts: dict[int, int] = {}
    for r in records:
        checks = r.diagnostics.get("sufficiency_checks", []) if r.diagnostics else []
        for c in checks:
            attempt = int(c.get("attempt", 0))
            retry_attempt_counts[attempt] = retry_attempt_counts.get(attempt, 0) + 1
            if c.get("used_llm"):
                llm_sufficiency_calls += 1
            else:
                heuristic_blocks += 1

    if retry_attempt_counts:
        ordered = ", ".join(f"a{a}={n}" for a, n in sorted(retry_attempt_counts.items()))
        print(f"Sufficiency checks by attempt: {ordered}")
        print(f"Sufficiency check mode      : heuristic={heuristic_blocks}, llm={llm_sufficiency_calls}")

    # Gate-vs-judge calibration quick view
    comparable = [r for r in records if r.gate_faithful is not None and r.judge_hallucination is not None]
    if comparable:
        agreements = 0
        gate_false_neg = 0
        gate_false_pos = 0
        for r in comparable:
            judge_faithful = not bool(r.judge_hallucination)
            if bool(r.gate_faithful) == judge_faithful:
                agreements += 1
            elif bool(r.gate_faithful) and not judge_faithful:
                gate_false_neg += 1
            elif (not bool(r.gate_faithful)) and judge_faithful:
                gate_false_pos += 1
        print(f"Gate/Judge agreement      : {agreements}/{len(comparable)}")
        print(f"Gate false negatives      : {gate_false_neg}")
        print(f"Gate false positives      : {gate_false_pos}")

    # Node error rate summary
    node_error_counts: dict[str, int] = {}
    for r in records:
        for node_name in (r.node_errors or {}).keys():
            node_error_counts[node_name] = node_error_counts.get(node_name, 0) + 1
    if node_error_counts:
        print("Node error rates:")
        for node_name, count in sorted(node_error_counts.items(), key=lambda kv: kv[1], reverse=True):
            print(f"  - {node_name}: {count}/{len(records)} ({count/len(records):.1%})")

    # Structured node error categories from diagnostics.
    category_counts: dict[str, int] = {}
    for r in records:
        events = (r.diagnostics or {}).get("node_error_events", [])
        for ev in events:
            category = ev.get("category", "unknown_error")
            category_counts[category] = category_counts.get(category, 0) + 1
    if category_counts:
        print("Node error categories:")
        for category, count in sorted(category_counts.items(), key=lambda kv: kv[1], reverse=True):
            print(f"  - {category}: {count}")

    # Retry ROI proxy: quality and source-hit payoff normalized by retry cost.
    retried_records = [r for r in records if r.retrieval_attempts > 0]
    if retried_records:
        helpful = sum(1 for r in retried_records if (r.judge_score or 0.0) >= 1.0 or r.retrieval_source_hits > 0)
        avg_attempts = sum(r.retrieval_attempts for r in retried_records) / len(retried_records)
        avg_judge = sum((r.judge_score or 0.0) for r in retried_records) / len(retried_records)
        score_per_attempt = avg_judge / (1.0 + avg_attempts)
        print(f"Retry ROI (proxy)      : helpful={helpful}/{len(retried_records)}, score/attempt={score_per_attempt:.3f}")

    # Retry effectiveness: first attempt index that hit any expected source reference.
    first_hit_attempts: list[int] = []
    for r in records:
        if not r.source_refs:
            continue
        expected = {ref.replace("\\", "/") for ref in r.source_refs if ref}
        retrieval_history = (r.diagnostics or {}).get("retrieval_history", [])
        first_hit: int | None = None
        for step in retrieval_history:
            refs = {ref.replace("\\", "/") for ref in step.get("source_refs", [])}
            if expected & refs:
                first_hit = int(step.get("attempt", 0))
                break
        if first_hit is not None:
            first_hit_attempts.append(first_hit)

    if first_hit_attempts:
        avg_first_hit = sum(first_hit_attempts) / len(first_hit_attempts)
        hit_on_initial = sum(1 for a in first_hit_attempts if a == 0)
        print(f"First source-hit attempt: avg={avg_first_hit:.2f}, attempt0={hit_on_initial}/{len(first_hit_attempts)}")

    # Breakdown by difficulty
    print(f"\n{'─'*75}")
    print(f"{'Level':<8} {'N':>4}  {'RAGAS':>7}  {'Judge':>7}  {'Halluc':>8}  {'Retries':>9}  {'Gate':>7}")
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

    # Difficulty-level retrieval source coverage breakdown.
    print(f"\n{'─'*75}")
    print(f"{'Level':<8} {'N':>4}  {'SrcRecall':>9}  {'AnySrcHit':>10}")
    print(f"{'─'*75}")
    for diff in ["L1", "L2", "L3", "L4"]:
        sub = [r for r in records if r.difficulty == diff]
        if not sub:
            continue
        recalls = [r.retrieval_source_recall for r in sub]
        any_hits = sum(1 for r in sub if r.retrieval_source_hits > 0)
        avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
        print(f"{diff:<8} {len(sub):>4}  {avg_recall:>9.3f}  {any_hits:>6}/{len(sub):<3}")
    print(f"{'='*75}")


##############################################################################
#                          RELOAD SAVED RECORDS                              #
##############################################################################

def load_records(path: str) -> list[AgenticEvalRecord]:
    return [AgenticEvalRecord(**item) for item in load_json(path)]


##############################################################################
#                                ENTRY POINT                                 #
##############################################################################

def evaluate_agentic_pipeline(
    benchmark_path: str = BENCHMARK_PATH,
    results_path: str   = RESULTS_PATH,
    skip_ragas: bool    = False,
    skip_judge: bool    = False,
    judge_only: bool    = False,
    max_questions: int | None = None,
    seed: int           = DEFAULT_SEED,
) -> list[AgenticEvalRecord]:
    validate_eval_flags(skip_ragas, skip_judge, judge_only)
    random.seed(seed)

    run_metadata = {
        "run_id": f"agentic-{uuid4()}",
        "run_started_at": utc_now_iso(),
        "generator_model": GENERATOR_MODEL,
        "evaluator_model": EVALUATOR_MODEL,
        "benchmark_path": benchmark_path,
        "seed": seed,
        "skip_ragas": skip_ragas,
        "skip_judge": skip_judge,
        "judge_only": judge_only,
        "max_questions": max_questions,
    }

    if judge_only:
        print(f"Loading existing results from {results_path}")
        records = load_records(results_path)
    else:
        benchmark = load_json(benchmark_path)
        validate_benchmark_items(benchmark)
        if max_questions is not None:
            benchmark = benchmark[:max_questions]

        records = run_agentic_pipeline(benchmark, run_metadata=run_metadata)

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
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    evaluate_agentic_pipeline(
        benchmark_path=args.benchmark,
        results_path=args.output,
        skip_ragas=args.skip_ragas,
        skip_judge=args.skip_judge,
        judge_only=args.judge_only,
        max_questions=args.max_questions,
        seed=args.seed,
    )
