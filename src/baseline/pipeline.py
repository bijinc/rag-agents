import os
from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv
from src.retrieval import Retriever, RetrievedChunk, SearchFilters
from src.constants import OPENROUTER_BASE_URL, MODELS, DEFAULT_MODEL

##############################################################################
#                              CONFIGURATION                                 #
##############################################################################

# load OpenRouter API key
load_dotenv()

DEFAULT_TOP_K = 5
MAX_TOKENS    = 1024

# System prompt — keeps the model grounded in retrieved context only
SYSTEM_PROMPT = """
You are a financial analyst assistant. Answer questions using ONLY the provided source passages.

Rules:
- If the answer is in the passages, state it clearly and cite which source it comes from (e.g. "According to AAPL 10-K 2024-11-01...").
- If the passages do not contain enough information to answer, say exactly: "The retrieved context does not contain sufficient information to answer this question."
- Do not use any knowledge outside the provided passages.
- Do not fabricate figures, dates, or management statements.
- For numerical answers, quote the exact figure from the source.
"""

##############################################################################
#                              DATA STRUCTURES                               #
##############################################################################

@dataclass
class PipelineResult:
    """Full result from one pipeline run — everything needed for evaluation."""
    question: str
    answer: str                          # model-generated answer
    retrieved_chunks: list[RetrievedChunk]
    contexts: list[str]                  # plain text of each chunk (for RAGAS)
    model: str
    top_k: int
    filters: SearchFilters | None = None

    @property
    def context_str(self) -> str:
        """Formatted context block exactly as passed to the model."""
        return _format_context(self.retrieved_chunks)
    

def _format_context(chunks: list[RetrievedChunk]) -> str:
    """
    Render retrieved chunks as a numbered context block for the prompt.
    Each chunk includes its source label so the model can cite it.
    """
    if not chunks:
        return "(no context retrieved)"

    parts = []
    for i, c in enumerate(chunks, 1):
        if c.source_type == "sec_filing":
            label = f"{c.ticker} {c.filing_type} ({c.filing_date})"
        else:
            label = f"{c.ticker} Earnings Call ({c.period})"
        parts.append(f"[{i}] {label}\n{c.text.strip()}")

    return "\n\n".join(parts)


def print_result(result: PipelineResult) -> None:
    """Pretty-print a PipelineResult for interactive use."""
    print(f"\n{'='*70}")
    print(f"Q: {result.question}")
    print(f"{'─'*70}")
    print(f"Model : {result.model}  |  top_k={result.top_k}")
    print(f"Retrieved {len(result.retrieved_chunks)} chunks:")
    for i, c in enumerate(result.retrieved_chunks, 1):
        src = (f"{c.ticker} {c.filing_type} {c.filing_date}"
               if c.source_type == "sec_filing"
               else f"{c.ticker} ECT {c.period}")
        print(f"  [{i}] score={c.score:.4f}  {src}")
    print(f"{'─'*70}")
    print(f"A: {result.answer}")
    print(f"{'='*70}\n")


##############################################################################
#                              PIPELINE                                      #
##############################################################################

class RAGPipeline:
    """
    Single-pass baseline RAG pipeline:
        question → hybrid retrieval → prompt → LLM → answer

    This is the baseline against which the agentic pipeline will be compared
    for RQ2. The same pipeline is used to generate answers for RQ1 (hallucination
    detection comparison between RAGAS and LLM-as-judge).
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        retriever: Retriever | None = None,
    ):
        """
        Args:
            model    : "qwen" | "llama" or a full OpenRouter model string
            retriever: optional pre-initialised Retriever (avoids reloading)
        """
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY not set.\n"
                "Export it in your shell:  export OPENROUTER_API_KEY=sk-or-..."
            )

        self._client    = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
        self._model     = MODELS.get(model, model)   # accept alias or full string
        self._retriever = retriever or Retriever()

    def run(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        filters: SearchFilters | None = None,
    ) -> PipelineResult:
        """
        Run one question through the full RAG pipeline.

        Args:
            question : natural language question
            top_k    : number of chunks to retrieve
            filters  : optional SearchFilters to scope retrieval

        Returns:
            PipelineResult containing answer + retrieved chunks + metadata
        """
        # Step 1 — Retrieve
        chunks = self._retriever.search(question, k=top_k, filters=filters)

        # Step 2 — Build prompt
        context_str  = _format_context(chunks)
        user_message = f"Source passages:\n\n{context_str}\n\nQuestion: {question}"

        # Step 3 — Generate
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
        )
        answer = response.choices[0].message.content.strip()

        return PipelineResult(
            question=question,
            answer=answer,
            retrieved_chunks=chunks,
            contexts=[c.text for c in chunks],
            model=self._model,
            top_k=top_k,
            filters=filters,
        )

    def run_batch(
        self,
        questions: list[str | dict],
        top_k: int = DEFAULT_TOP_K,
    ) -> list[PipelineResult]:
        """
        Run a list of questions. Each item can be:
          - a plain string (no filters)
          - a dict with keys "question" and optionally "filters"

        Returns results in the same order.
        """
        results = []
        for i, item in enumerate(questions, 1):
            if isinstance(item, str):
                question, filters = item, None
            else:
                question = item["question"]
                filters  = item.get("filters")

            print(f"  [{i:2d}/{len(questions)}] {question[:70]}")
            result = self.run(question, top_k=top_k, filters=filters)
            results.append(result)

        return results


def test_pipeline():
    # Tests one question from each difficulty tier
    pipeline = RAGPipeline(model=DEFAULT_MODEL)

    test_questions = [
        # L1 — direct extraction
        "What was Apple's total net revenue for fiscal year 2024?",
        # L2 — cross-period
        "How did Apple's gross margin change between Q3 2023 and Q3 2024?",
        # L3 — management commentary
        "What did Apple's management say about the outlook for Services revenue growth in the Q3 2024 earnings call?",
        # L4 — multi-source
        "Apple's 10-K disclosed a significant increase in R&D spending in FY2024 — did management address the rationale for this in the earnings call?",
    ]

    for q in test_questions:
        result = pipeline.run(q, top_k=5)
        print_result(result)
