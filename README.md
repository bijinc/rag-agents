# rag-agents

Baseline and agentic RAG evaluation harness for financial QA.

## Quick Start

Install uv

[Documentation](https://docs.astral.sh/uv/getting-started/installation/)

```
pip install uv
```

1. Create and sync environment

```bash
uv sync
source .venv/bin/activate
```

2. Set required environment variables

```bash
cp .env.example .env
# then edit .env and set OPENROUTER_API_KEY
```

3. Build index and run both evaluations

```bash
python main.py
```

## Evaluation Commands

Run complete pipeline from scratch. Downloads data and builds index before running pipelines.

```bash
python main.py --latest --reindex
```

Run baseline harness:

```bash
python -m src.baseline.evaluation --max-questions 5 --seed 42
```

Run agentic harness:

```bash
python -m src.agentic.agentic_evaluation --max-questions 5 --seed 42
```

Run comparative summary + chart generation from existing result files:

```bash
python -m src.evaluation_summary --baseline results/baseline_eval_results.json --agentic results/agentic_eval_results.json
```

Useful flags:

- `--latest`
- `--reindex`
- `--max-questions`
- `--seed`

## Output Artifacts

- Baseline results: `results/baseline_eval_results.json`
- Agentic results: `results/agentic_eval_results.json`
- Comparative charts: `results/eval_summary_charts/`
