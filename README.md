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

Run baseline harness:

```bash
python -m src.baseline.evaluation --max-questions 5 --seed 42
```

Run agentic harness:

```bash
python -m src.agentic.agentic_evaluation --max-questions 5 --seed 42
```

Useful flags:

- `--skip-ragas`
- `--skip-judge`
- `--judge-only`
- `--max-questions`
- `--seed`

## Output Artifacts

- Baseline results: `data/baseline_eval_results.json`
- Agentic results: `data/agentic_eval_results.json`
- Baseline metadata: `data/baseline_eval_results.json.meta`
