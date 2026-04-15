# rag-agents

Baseline and agentic RAG evaluation harness for financial QA.

## Project Structure

```text
rag-agents/
├── main.py                          # Entry point: builds index + runs evaluations
├── pyproject.toml                   # Project dependencies and tooling config
├── qa_benchmark.json                # Benchmark questions/ground truth
├── README.md
├── data/
│   ├── raw/
│   │   ├── ect/                     # Parsed earnings call transcript data
│   │   └── sec_filings/             # Parsed SEC filing data
│   └── chroma_db/                   # Local vector index persistence
└── src/
    ├── chunking.py                  # Text chunking logic
    ├── constants.py                 # Shared constants/config values
    ├── data.py                      # Data loading and preprocessing
    ├── eval_utils.py                # Shared evaluation helpers
    ├── finance.py                   # Finance-domain helpers
    ├── retrieval.py                 # Retriever setup and search
    ├── baseline/
    │   ├── evaluation.py            # Baseline evaluation harness
    │   └── pipeline.py              # Baseline RAG pipeline
    └── agentic/
        ├── agentic_evaluation.py    # Agentic evaluation harness
        ├── agentic_pipeline.py      # Agentic orchestration pipeline
        └── nodes/
            ├── faithfulness_gate.py
            ├── generator_node.py
            ├── llm_utils.py
            ├── node_schemas.py
            ├── query_analyzer.py
            ├── query_refiner.py
            ├── retriever_node.py
            └── sufficiency_checker.py
```

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
