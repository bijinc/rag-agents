import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns

from src.constants import AGENTIC_RESULTS_PATH, BASELINE_RESULTS_PATH, SUMMARY_CHARTS_DIR, SUMMARY_CHARTS_PREFIX
from src.utils.eval_utils import load_json


DIFFICULTIES = ("L1", "L2", "L3", "L4")


@dataclass
class SummaryRow:
    pipeline: str
    question_id: str
    question: str
    difficulty: str
    ragas_score: float | None
    ragas_status: str
    judge_score: float | None
    judge_status: str
    judge_hallucination: bool | None
    judge_hallucination_type: str | None


class EvaluationSummary:
    """Summarize baseline + agentic evaluation artifacts with chart-ready outputs."""

    def __init__(
        self,
        baseline_path: str = BASELINE_RESULTS_PATH,
        agentic_path: str = AGENTIC_RESULTS_PATH,
    ):
        self.baseline_path = Path(baseline_path)
        self.agentic_path = Path(agentic_path)
        self.rows: list[SummaryRow] = []

    @classmethod
    def from_paths(
        cls,
        baseline_path: str = BASELINE_RESULTS_PATH,
        agentic_path: str = AGENTIC_RESULTS_PATH,
    ) -> "EvaluationSummary":
        summary = cls(baseline_path=baseline_path, agentic_path=agentic_path)
        summary.load()
        return summary

    @classmethod
    def from_records(
        cls,
        baseline_records: list[Any] | None = None,
        agentic_records: list[Any] | None = None,
    ) -> "EvaluationSummary":
        """Build a summary directly from in-memory pipeline records."""
        summary = cls()
        summary.rows = []
        summary.rows.extend(summary._rows_from_records(baseline_records or [], "baseline"))
        summary.rows.extend(summary._rows_from_records(agentic_records or [], "agentic"))
        return summary

    def load(self) -> None:
        self.rows = []
        self.rows.extend(self._load_pipeline_rows(self.baseline_path, "baseline"))
        self.rows.extend(self._load_pipeline_rows(self.agentic_path, "agentic"))

    def _load_pipeline_rows(self, path: Path, pipeline: str) -> list[SummaryRow]:
        if not path.exists():
            return []

        payload = load_json(str(path))
        if not isinstance(payload, list):
            raise ValueError(f"Expected list payload in {path}, got {type(payload).__name__}")

        rows: list[SummaryRow] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            rows.append(
                SummaryRow(
                    pipeline=pipeline,
                    question_id=str(item.get("id", "")),
                    question=str(item.get("question", "")),
                    difficulty=self._normalize_difficulty(item.get("difficulty")),
                    ragas_score=self._to_float_or_none(item.get("ragas_faithfulness")),
                    ragas_status=str(item.get("ragas_status", "not_run")),
                    judge_score=self._to_float_or_none(item.get("judge_score")),
                    judge_status=str(item.get("judge_status", "not_run")),
                    judge_hallucination=self._to_bool_or_none(item.get("judge_hallucination")),
                    judge_hallucination_type=item.get("judge_hallucination_type"),
                )
            )
        return rows

    def _rows_from_records(self, records: list[Any], pipeline: str) -> list[SummaryRow]:
        rows: list[SummaryRow] = []
        for record in records:
            if isinstance(record, dict):
                item = record
            else:
                item = vars(record)
            rows.append(
                SummaryRow(
                    pipeline=pipeline,
                    question_id=str(item.get("id", "")),
                    question=str(item.get("question", "")),
                    difficulty=self._normalize_difficulty(item.get("difficulty")),
                    ragas_score=self._to_float_or_none(item.get("ragas_faithfulness")),
                    ragas_status=str(item.get("ragas_status", "not_run")),
                    judge_score=self._to_float_or_none(item.get("judge_score")),
                    judge_status=str(item.get("judge_status", "not_run")),
                    judge_hallucination=self._to_bool_or_none(item.get("judge_hallucination")),
                    judge_hallucination_type=item.get("judge_hallucination_type"),
                )
            )
        return rows

    @staticmethod
    def _to_float_or_none(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_bool_or_none(value: Any) -> bool | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.lower().strip()
            if lowered in {"true", "1", "yes"}:
                return True
            if lowered in {"false", "0", "no"}:
                return False
        if isinstance(value, (int, float)):
            return bool(value)
        return None

    @staticmethod
    def _normalize_difficulty(value: Any) -> str:
        text = str(value or "").upper().strip()
        return text if text in DIFFICULTIES else "UNKNOWN"

    def is_empty(self) -> bool:
        return len(self.rows) == 0

    def _pipelines(self) -> list[str]:
        return sorted({row.pipeline for row in self.rows})

    def _rows_for(self, pipeline: str, difficulty: str) -> list[SummaryRow]:
        return [r for r in self.rows if r.pipeline == pipeline and r.difficulty == difficulty]

    def _mean(self, values: list[float]) -> float | None:
        if not values:
            return None
        return sum(values) / len(values)

    def aggregate_by_question_type(self) -> list[dict[str, Any]]:
        """Return L1-L4 aggregates for each pipeline (chart-ready)."""
        output: list[dict[str, Any]] = []

        for pipeline in self._pipelines():
            for diff in DIFFICULTIES:
                subset = self._rows_for(pipeline, diff)
                n_total = len(subset)
                ragas_values = [r.ragas_score for r in subset if r.ragas_score is not None]
                judge_values = [r.judge_score for r in subset if r.judge_score is not None]
                hallucinated = [r for r in subset if r.judge_hallucination is True]
                halluc_rate = (len(hallucinated) / n_total) if n_total else None
                halluc_types: dict[str, int] = {}
                for row in hallucinated:
                    key = row.judge_hallucination_type or "unspecified"
                    halluc_types[key] = halluc_types.get(key, 0) + 1

                output.append(
                    {
                        "pipeline": pipeline,
                        "difficulty": diff,
                        "n_total": n_total,
                        "ragas_mean": self._mean(ragas_values),
                        "ragas_min": min(ragas_values) if ragas_values else None,
                        "ragas_max": max(ragas_values) if ragas_values else None,
                        "ragas_completion_rate": (len(ragas_values) / n_total) if n_total else None,
                        "judge_mean": self._mean(judge_values),
                        "judge_min": min(judge_values) if judge_values else None,
                        "judge_max": max(judge_values) if judge_values else None,
                        "judge_completion_rate": (len(judge_values) / n_total) if n_total else None,
                        "hallucination_count": len(hallucinated),
                        "hallucination_rate": halluc_rate,
                        "hallucination_types": halluc_types,
                    }
                )

        return output

    def aggregate_overall(self) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for pipeline in self._pipelines():
            subset = [r for r in self.rows if r.pipeline == pipeline]
            n_total = len(subset)
            ragas_values = [r.ragas_score for r in subset if r.ragas_score is not None]
            judge_values = [r.judge_score for r in subset if r.judge_score is not None]
            halluc_count = sum(1 for r in subset if r.judge_hallucination is True)
            output.append(
                {
                    "pipeline": pipeline,
                    "n_total": n_total,
                    "ragas_mean": self._mean(ragas_values),
                    "judge_mean": self._mean(judge_values),
                    "hallucination_rate": (halluc_count / n_total) if n_total else None,
                }
            )
        return output

    def print_text_summary(self) -> None:
        rows = self.aggregate_by_question_type()
        if not rows:
            print("No records found in provided evaluation files.")
            return

        print("\n" + "=" * 86)
        print("COMPARATIVE EVALUATION SUMMARY (by question type / difficulty)")
        print("=" * 86)
        print(f"{'Pipeline':<10} {'Type':<4} {'N':>4}  {'RAGAS':>7}  {'Judge':>7}  {'Halluc':>8}")
        print("-" * 86)

        for pipeline in sorted({r["pipeline"] for r in rows}):
            for row in [r for r in rows if r["pipeline"] == pipeline]:
                ragas = f"{row['ragas_mean']:.3f}" if row["ragas_mean"] is not None else "  —"
                judge = f"{row['judge_mean']:.3f}" if row["judge_mean"] is not None else "  —"
                hc = f"{row['hallucination_count']}/{row['n_total']}"
                print(
                    f"{pipeline:<10} {row['difficulty']:<4} {row['n_total']:>4}  "
                    f"{ragas:>7}  {judge:>7}  {hc:>8}"
                )

        print("=" * 86)

    def save_charts(self, output_dir: str = SUMMARY_CHARTS_DIR, prefix: str = SUMMARY_CHARTS_PREFIX) -> list[str]:
        """Generate static PNG charts and return saved paths."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        summary = self.aggregate_by_question_type()
        if not summary:
            return []

        sns.set_theme(style="whitegrid")

        chart_paths: list[str] = []
        chart_paths.append(self._chart_grouped_metric(summary, "ragas_mean", "Mean RAGAS Faithfulness", output_dir, prefix))
        chart_paths.append(self._chart_grouped_metric(summary, "judge_mean", "Mean LLM Judge Score", output_dir, prefix))
        chart_paths.append(self._chart_grouped_metric(summary, "hallucination_rate", "Hallucination Rate", output_dir, prefix, y_max=1.0))
        chart_paths.append(self._chart_completion_rates(summary, output_dir, prefix))
        chart_paths.append(self._chart_hallucination_types(summary, output_dir, prefix))

        return [p for p in chart_paths if p]

    def _chart_grouped_metric(
        self,
        summary_rows: list[dict[str, Any]],
        metric_key: str,
        title: str,
        output_dir: str,
        prefix: str,
        y_max: float | None = None,
    ) -> str:
        pipelines = sorted({r["pipeline"] for r in summary_rows})
        x_positions = list(range(len(DIFFICULTIES)))
        width = 0.38 if len(pipelines) > 1 else 0.55

        fig, ax = plt.subplots(figsize=(10, 5))
        for i, pipeline in enumerate(pipelines):
            offsets = [x + (i - (len(pipelines) - 1) / 2) * width for x in x_positions]
            vals = []
            for diff in DIFFICULTIES:
                row = next((r for r in summary_rows if r["pipeline"] == pipeline and r["difficulty"] == diff), None)
                val = row.get(metric_key) if row else None
                vals.append(0.0 if val is None else float(val))
            ax.bar(offsets, vals, width=width, label=pipeline)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(DIFFICULTIES)
        ax.set_ylabel(metric_key)
        ax.set_title(title)
        if y_max is not None:
            ax.set_ylim(0.0, y_max)
        ax.legend()
        fig.tight_layout()

        out_path = str(Path(output_dir) / f"{prefix}_{metric_key}.png")
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        return out_path

    def _chart_completion_rates(self, summary_rows: list[dict[str, Any]], output_dir: str, prefix: str) -> str:
        pipelines = sorted({r["pipeline"] for r in summary_rows})
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

        for ax, metric_key, title in [
            (axes[0], "ragas_completion_rate", "RAGAS Completion Rate"),
            (axes[1], "judge_completion_rate", "Judge Completion Rate"),
        ]:
            for pipeline in pipelines:
                vals = []
                for diff in DIFFICULTIES:
                    row = next((r for r in summary_rows if r["pipeline"] == pipeline and r["difficulty"] == diff), None)
                    vals.append(float(row.get(metric_key) or 0.0) if row else 0.0)
                ax.plot(DIFFICULTIES, vals, marker="o", label=pipeline)
            ax.set_title(title)
            ax.set_ylim(0.0, 1.05)
            ax.set_xlabel("Question type")

        axes[0].set_ylabel("rate")
        axes[1].legend()
        fig.tight_layout()

        out_path = str(Path(output_dir) / f"{prefix}_completion_rates.png")
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        return out_path

    def _chart_hallucination_types(self, summary_rows: list[dict[str, Any]], output_dir: str, prefix: str) -> str:
        labels: list[str] = []
        counts: list[int] = []

        for row in summary_rows:
            pipeline = row["pipeline"]
            diff = row["difficulty"]
            for h_type, count in sorted(row["hallucination_types"].items()):
                labels.append(f"{pipeline}-{diff}-{h_type}")
                counts.append(count)

        fig, ax = plt.subplots(figsize=(12, 5))
        if counts:
            positions = list(range(len(labels)))
            ax.bar(positions, counts)
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=35, ha="right")
        else:
            ax.text(0.5, 0.5, "No hallucination types recorded", ha="center", va="center")
            ax.set_xticks([])

        ax.set_title("Hallucination Type Counts (pipeline-difficulty)")
        ax.set_ylabel("count")
        fig.tight_layout()

        out_path = str(Path(output_dir) / f"{prefix}_hallucination_types.png")
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        return out_path


def run_summary(
    baseline_path: str = BASELINE_RESULTS_PATH,
    agentic_path: str = AGENTIC_RESULTS_PATH,
    output_dir: str = SUMMARY_CHARTS_DIR,
    prefix: str = SUMMARY_CHARTS_PREFIX,
) -> tuple[EvaluationSummary, list[str]]:
    summary = EvaluationSummary.from_paths(baseline_path=baseline_path, agentic_path=agentic_path)
    summary.print_text_summary()
    chart_paths = summary.save_charts(output_dir=output_dir, prefix=prefix)
    return summary, chart_paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize baseline + agentic evaluation outputs")
    parser.add_argument("--baseline", default=BASELINE_RESULTS_PATH, help="Path to baseline eval results JSON")
    parser.add_argument("--agentic", default=AGENTIC_RESULTS_PATH, help="Path to agentic eval results JSON")
    parser.add_argument("--output-dir", default=SUMMARY_CHARTS_DIR, help="Directory where chart PNG files will be written")
    parser.add_argument("--prefix", default=SUMMARY_CHARTS_PREFIX, help="Prefix for generated chart filenames")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    _, saved = run_summary(
        baseline_path=args.baseline,
        agentic_path=args.agentic,
        output_dir=args.output_dir,
        prefix=args.prefix,
    )
    if saved:
        print("\nSaved chart artifacts:")
        for path in saved:
            print(f"  - {path}")
    else:
        print("\nNo charts generated. Ensure at least one results file exists with records.")
