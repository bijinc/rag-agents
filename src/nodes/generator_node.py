"""src/nodes/generator_node.py

LangGraph node: Generator

Formats retrieved chunks into a numbered context block and calls the
LLM to produce an answer. Uses the same grounding prompt as the
baseline RAGPipeline to ensure a fair comparison.
"""

from openai import OpenAI
from src.retrieval import RetrievedChunk

##############################################################################
#                              CONFIGURATION                                 #
##############################################################################

# Same system prompt as src/pipeline.py — ensures fair baseline comparison
SYSTEM_PROMPT = """You are a financial analyst assistant. Answer questions using ONLY the provided source passages.

Rules:
- If the answer is in the passages, state it clearly and cite which source it comes from (e.g. "According to AAPL 10-K 2024-11-01...").
- If the passages do not contain enough information to answer, say exactly: "The retrieved context does not contain sufficient information to answer this question."
- Do not use any knowledge outside the provided passages.
- Do not fabricate figures, dates, or management statements.
- For numerical answers, quote the exact figure from the source."""

MAX_TOKENS = 1024


##############################################################################
#                              HELPERS                                       #
##############################################################################

def _format_context(chunks: list[RetrievedChunk]) -> str:
    """Numbered context block — same format as baseline pipeline."""
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


##############################################################################
#                              NODE                                          #
##############################################################################

def generator_node(state: dict, client: OpenAI, model: str) -> dict:
    """
    Generate an answer from retrieved chunks using the grounded system prompt.

    Returns updates for: answer
    """
    question = state["question"]
    chunks   = state.get("retrieved_chunks", [])

    context_str  = _format_context(chunks)
    user_message = f"Source passages:\n\n{context_str}\n\nQuestion: {question}"

    response = client.chat.completions.create(
        model=model,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
    )
    answer = response.choices[0].message.content.strip()

    print(f"  [generator] Generated {len(answer)} chars using {len(chunks)} chunks")

    return {"answer": answer}
