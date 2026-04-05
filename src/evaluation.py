"""src/evaluation.py

RQ1 evaluation: RAGAS faithfulness vs LLM-as-judge hallucination detection.

Generator  : qwen/qwen-2.5-7b-instruct        (via OpenRouter)
Evaluator  : meta-llama/llama-3.1-8b-instruct  (via OpenRouter, different from generator)

Usage:
    python -m src.evaluation
    python -m src.evaluation --skip-ragas       # judge only
    python -m src.evaluation --skip-judge       # RAGAS only
"""

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from src.pipeline import RAGPipeline, PipelineResult

##############################################################################
#                              CONFIGURATION                                 #
##############################################################################

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

GENERATOR_MODEL = "qwen/qwen-2.5-7b-instruct"
EVALUATOR_MODEL = "meta-llama/llama-3.1-8b-instruct"

BENCHMARK_PATH = "data/qa_benchmark.json"
RESULTS_PATH   = "data/eval_results.json"
TOP_K          = 5

##############################################################################
#                           DATA STRUCTURES                                  #
##############################################################################

@dataclass
class EvalRecord:
    """One benchmark question fully evaluated by both methods."""
    id: str
    question: str
    difficulty: str
    tickers: list[str]
    ground_truth: str
    generated_answer: str
    contexts: list[str]
    source_refs: list[str]
    # RAGAS
    ragas_faithfulness: float | None = None
    # LLM-as-judge
    judge_score: float | None = None
    judge_hallucination: bool | None = None
    judge_hallucination_type: str | None = None
    judge_reasoning: str | None = None
    # metadata
    generator_model: str = GENERATOR_MODEL
    evaluator_model: str = EVALUATOR_MODEL
    top_k: int = TOP_K


##############################################################################
#                        STEP 1 — RUN PIPELINE                              #
##############################################################################

def run_pipeline(
    benchmark_path: str = BENCHMARK_PATH,top_k: int = TOP_K,) -> list[EvalRecord]:
    """Run the RAG pipeline on every benchmark question."""
    with open(benchmark_path) as f:
        benchmark = json.load(f)

    pipeline = RAGPipeline(model=GENERATOR_MODEL)
    records: list[EvalRecord] = []

    print(f"\nRunning pipeline on {len(benchmark)} questions  [generator: {GENERATOR_MODEL}]")
    for i, item in enumerate(benchmark, 1):
        print(f"  [{i:2d}/{len(benchmark)}] {item['question'][:70]}")
        result: PipelineResult = pipeline.run(item["question"], top_k=top_k)
        records.append(EvalRecord(
            id=item["id"],
            question=item["question"],
            difficulty=item["difficulty"],
            tickers=item.get("tickers", []),
            ground_truth=item["answer"],
            generated_answer=result.answer,
            contexts=result.contexts,
            source_refs=item.get("source_refs", []),
        ))
        time.sleep(0.2)

    return records


##############################################################################
#                      STEP 2 — RAGAS FAITHFULNESS                          #
##############################################################################

def run_ragas(records: list[EvalRecord]) -> None:
    """Compute RAGAS faithfulness score in-place on each EvalRecord."""
    import warnings
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.metrics._faithfulness import Faithfulness  # legacy path: is a Metric subclass
    from ragas.llms import LangchainLLMWrapper             # deprecated but correct type for legacy metrics
    from langchain_openai import ChatOpenAI

    print(f"\nRunning RAGAS faithfulness  [evaluator: {EVALUATOR_MODEL}]")

    lc_llm = ChatOpenAI(
        model=EVALUATOR_MODEL,
        base_url=OPENROUTER_BASE_URL,
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        ragas_llm = LangchainLLMWrapper(lc_llm)
    metric = Faithfulness(llm=ragas_llm)

    samples = [
        SingleTurnSample(
            user_input=r.question,
            response=r.generated_answer,
            retrieved_contexts=r.contexts,
        )
        for r in records
    ]
    dataset = EvaluationDataset(samples=samples)

    result = evaluate(
        dataset=dataset,
        metrics=[metric],
        raise_exceptions=False,
        show_progress=True,
    )
    scores = result.to_pandas()["faithfulness"].tolist()

    for record, score in zip(records, scores):
        record.ragas_faithfulness = (
            round(float(score), 4) if score is not None else None
        )

    valid = [s for s in scores if s is not None]
    print(f"  Done — mean faithfulness: {sum(valid)/len(valid):.3f}" if valid else "  All scores were None")


##############################################################################
#                      STEP 3 — LLM-AS-JUDGE                               #
##############################################################################

_JUDGE_SYSTEM = """You are a financial analyst evaluating whether an AI-generated answer contains hallucinations compared to a known-correct ground truth.

Rules:
1. If the generated answer is a refusal (contains "does not contain sufficient information" or similar) — it made no claims. Assign score=0, hallucination_detected=false, hallucination_type=null, reasoning="Refusal — no answer attempted."
2. If the generated answer makes claims that are ALL consistent with the ground truth — score=1, hallucination_detected=false.
3. If the generated answer makes ANY claim that contradicts or is absent from the ground truth — score=0, hallucination_detected=true.

Hallucination types (use only when hallucination_detected=true):
- "fabricated_figure"   : wrong number, percentage, or monetary value
- "temporal_confusion"  : correct fact but wrong time period
- "altered_guidance"    : management statement misquoted or distorted
- "multi_source"        : claim requires combining sources but is stated without support

Output a JSON object with exactly these keys:
  "score"                  : integer 0 or 1
  "hallucination_detected" : boolean
  "hallucination_type"     : one of the four types above, or null
  "reasoning"              : one sentence naming the specific claim that is wrong (or "Refusal" or "Faithful")

Output ONLY valid JSON. No markdown fences, no prose."""

_JUDGE_USER = """QUESTION:
{question}

GROUND TRUTH:
{ground_truth}

GENERATED ANSWER:
{generated_answer}"""


def run_llm_judge(records: list[EvalRecord]) -> None:
    """Run LLM-as-judge evaluation in-place on each EvalRecord."""
    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
    )
    print(f"\nRunning LLM-as-judge  [evaluator: {EVALUATOR_MODEL}]")

    for i, record in enumerate(records, 1):
        print(f"  [{i:2d}/{len(records)}] {record.question[:70]}")
        user_msg = _JUDGE_USER.format(
            question=record.question,
            ground_truth=record.ground_truth,
            generated_answer=record.generated_answer,
        )
        try:
            response = client.chat.completions.create(
                model=EVALUATOR_MODEL,
                max_tokens=300,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
            )
            raw = response.choices[0].message.content.strip()
            # Strip markdown code fences if the model adds them
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.rstrip("`").strip()
            verdict = json.loads(raw)
            record.judge_score              = round(float(verdict.get("score", 0.0)), 4)
            record.judge_hallucination      = bool(verdict.get("hallucination_detected", False))
            record.judge_hallucination_type = verdict.get("hallucination_type")
            record.judge_reasoning          = verdict.get("reasoning", "")
        except Exception as e:
            print(f"    WARNING: judge failed for id={record.id}: {e}")
            record.judge_score         = None
            record.judge_hallucination = None
        time.sleep(0.2)

    detected = sum(1 for r in records if r.judge_hallucination)
    print(f"  Done — hallucinations detected: {detected}/{len(records)}")


##############################################################################
#                          SAVE + SUMMARY                                    #
##############################################################################

def save_results(records: list[EvalRecord], path: str = RESULTS_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([asdict(r) for r in records], f, indent=2)
    print(f"\nResults saved → {path}")


def print_summary(records: list[EvalRecord]) -> None:
    print(f"\n{'='*65}")
    print(f"EVALUATION SUMMARY  ({len(records)} questions)")
    print(f"  Generator : {GENERATOR_MODEL}")
    print(f"  Evaluator : {EVALUATOR_MODEL}")
    print(f"{'='*65}")

    ragas_scores  = [r.ragas_faithfulness for r in records if r.ragas_faithfulness is not None]
    judge_scores  = [r.judge_score        for r in records if r.judge_score        is not None]
    hallucinated  = sum(1 for r in records if r.judge_hallucination)

    def fmt(scores: list[float]) -> str:
        if not scores:
            return "  n/a"
        return f"mean={sum(scores)/len(scores):.3f}  min={min(scores):.3f}  max={max(scores):.3f}"

    print(f"RAGAS Faithfulness    : {fmt(ragas_scores)}")
    print(f"LLM Judge Score       : {fmt(judge_scores)}")
    print(f"Hallucinations (judge): {hallucinated}/{len(records)}")

    # Breakdown by difficulty
    print(f"\n{'─'*65}")
    print(f"{'Level':<8} {'N':>4}  {'RAGAS':>7}  {'Judge':>7}  {'Halluc':>8}")
    print(f"{'─'*65}")
    for diff in ["L1", "L2", "L3", "L4"]:
        sub = [r for r in records if r.difficulty == diff]
        if not sub:
            continue
        rs = [r.ragas_faithfulness for r in sub if r.ragas_faithfulness is not None]
        js = [r.judge_score        for r in sub if r.judge_score        is not None]
        hc = sum(1 for r in sub if r.judge_hallucination)
        r_mean = f"{sum(rs)/len(rs):.3f}" if rs else "  —"
        j_mean = f"{sum(js)/len(js):.3f}" if js else "  —"
        print(f"{diff:<8} {len(sub):>4}  {r_mean:>7}  {j_mean:>7}  {hc:>4}/{len(sub)}")
    print(f"{'='*65}")


##############################################################################
#                                ENTRY POINT                                 #
##############################################################################

def load_records(path: str) -> list[EvalRecord]:
    """Reload saved EvalRecords from a previous run."""
    with open(path) as f:
        return [EvalRecord(**item) for item in json.load(f)]


def evaluate_pipeline(
    benchmark_path: str = BENCHMARK_PATH,
    results_path: str   = RESULTS_PATH,
    top_k: int          = TOP_K,
    skip_ragas: bool    = False,
    skip_judge: bool    = False,
    judge_only: bool    = False,
) -> list[EvalRecord]:
    if judge_only:
        print(f"Loading existing results from {results_path}")
        records = load_records(results_path)
    else:
        records = run_pipeline(benchmark_path, top_k=top_k)
        if not skip_ragas:
            run_ragas(records)

    if not skip_judge:
        run_llm_judge(records)

    save_results(records, results_path)
    print_summary(records)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG evaluation pipeline")
    parser.add_argument("--skip-ragas",  action="store_true", help="Skip RAGAS faithfulness step")
    parser.add_argument("--skip-judge",  action="store_true", help="Skip LLM-as-judge step")
    parser.add_argument("--judge-only",  action="store_true", help="Load existing results and run only the judge")
    parser.add_argument("--top-k",       type=int, default=TOP_K)
    parser.add_argument("--benchmark",   default=BENCHMARK_PATH)
    parser.add_argument("--output",      default=RESULTS_PATH)
    args = parser.parse_args()

    evaluate_pipeline(
        benchmark_path=args.benchmark,
        results_path=args.output,
        top_k=args.top_k,
        skip_ragas=args.skip_ragas,
        skip_judge=args.skip_judge,
        judge_only=args.judge_only,
    )
