"""src/agentic_pipeline.py

Agentic RAG pipeline using LangGraph.

Compared against the single-pass RAGPipeline (src/pipeline.py) for RQ2:
"Does agentic retrieval reduce hallucination rates?"

Graph:
    START
      → query_analyzer       (decompose question, extract filters)
      → retriever            (hybrid Dense + BM25 + RRF search)
      → sufficiency_checker  (is the context good enough?)
          → [insufficient + retries remain] → query_refiner → retriever
          → [sufficient | retries exhausted] → generator
      → generator            (generate grounded answer)
      → faithfulness_gate    (flag answers not grounded in context)
      → END

Usage:
    python -m src.agentic_pipeline
"""

import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

load_dotenv()

from src.retrieval import Retriever, RetrievedChunk, SearchFilters
from src.agentic.nodes.query_analyzer    import query_analyzer_node
from src.agentic.nodes.retriever_node    import retriever_node
from src.agentic.nodes.sufficiency_checker import sufficiency_checker_node, route_sufficiency
from src.agentic.nodes.query_refiner     import query_refiner_node
from src.agentic.nodes.generator_node    import generator_node
from src.agentic.nodes.faithfulness_gate import faithfulness_gate_node

##############################################################################
#                              CONFIGURATION                                 #
##############################################################################

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

MODELS = {
    "qwen":  "qwen/qwen-2.5-7b-instruct",
    "llama": "meta-llama/llama-3-8b-instruct",
}
DEFAULT_MODEL = "qwen"

##############################################################################
#                              LANGGRAPH STATE                               #
##############################################################################

class AgentState(TypedDict):
    # Input
    question:             str
    original_question:    str
    # Query decomposition
    sub_questions:        list[str]
    filters:              Any          # SearchFilters | None
    question_type:        str          # "L1" | "L2" | "L3" | "L4"
    # Retrieval
    retrieved_chunks:     list[Any]    # list[RetrievedChunk]
    retrieval_attempts:   int
    sufficiency:          str          # "sufficient" | "insufficient"
    # Generation
    answer:               str
    # Faithfulness
    is_faithful:          bool
    confidence:           str          # "high" | "low"
    faithfulness_reasoning: str | None


##############################################################################
#                              RESULT TYPE                                   #
##############################################################################

@dataclass
class AgenticResult:
    """Full result from one agentic pipeline run."""
    question:               str
    answer:                 str
    sub_questions:          list[str]
    retrieved_chunks:       list[RetrievedChunk]
    contexts:               list[str]     # plain text per chunk (for RAGAS)
    retrieval_attempts:     int
    is_faithful:            bool
    confidence:             str
    faithfulness_reasoning: str | None
    question_type:          str
    model:                  str


##############################################################################
#                              PIPELINE CLASS                                #
##############################################################################

class AgenticPipeline:
    """
    Agentic RAG pipeline with:
      - Query decomposition (query_analyzer)
      - Retrieval sufficiency checking with up to 2 retries (sufficiency_checker + query_refiner)
      - Faithfulness gating on the generated answer (faithfulness_gate)

    The baseline pipeline (src/pipeline.py) runs the same retrieval and
    generation without any of these agentic layers.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        retriever: Retriever | None = None,
    ):
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY not set.\n"
                "Export it:  export OPENROUTER_API_KEY=sk-or-..."
            )

        self._client    = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
        self._model     = MODELS.get(model, model)
        self._retriever = retriever or Retriever()
        self._graph     = self._build_graph()

    # -------------------------------------------------------------------------
    # Graph construction
    # -------------------------------------------------------------------------

    def _build_graph(self):
        """Wire up the LangGraph StateGraph, binding shared resources via closures."""

        # Each closure captures self._client / self._retriever / self._model
        def _query_analyzer(state):
            return query_analyzer_node(state, self._client)

        def _retriever(state):
            return retriever_node(state, self._retriever)

        def _sufficiency_checker(state):
            return sufficiency_checker_node(state, self._client)

        def _query_refiner(state):
            return query_refiner_node(state, self._client)

        def _generator(state):
            return generator_node(state, self._client, self._model)

        def _faithfulness_gate(state):
            return faithfulness_gate_node(state, self._client)

        # Build and connect the graph
        graph = StateGraph(AgentState)

        graph.add_node("query_analyzer",     _query_analyzer)
        graph.add_node("retriever",          _retriever)
        graph.add_node("sufficiency_checker",_sufficiency_checker)
        graph.add_node("query_refiner",      _query_refiner)
        graph.add_node("generator",          _generator)
        graph.add_node("faithfulness_gate",  _faithfulness_gate)

        graph.add_edge(START,                "query_analyzer")
        graph.add_edge("query_analyzer",     "retriever")
        graph.add_edge("retriever",          "sufficiency_checker")
        graph.add_conditional_edges(
            "sufficiency_checker",
            route_sufficiency,
            {
                "sufficient":   "generator",
                "insufficient": "query_refiner",
            },
        )
        graph.add_edge("query_refiner",      "retriever")
        graph.add_edge("generator",          "faithfulness_gate")
        graph.add_edge("faithfulness_gate",  END)

        return graph.compile()

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def run(self, question: str) -> AgenticResult:
        """Run one question through the full agentic pipeline."""
        print(f"\n{'='*65}")
        print(f"[agentic] Q: {question[:70]}")
        print(f"{'─'*65}")

        initial_state: AgentState = {
            "question":              question,
            "original_question":     question,
            "sub_questions":         [],
            "filters":               None,
            "question_type":         "L1",
            "retrieved_chunks":      [],
            "retrieval_attempts":    0,
            "sufficiency":           "sufficient",
            "answer":                "",
            "is_faithful":           True,
            "confidence":            "high",
            "faithfulness_reasoning": None,
        }

        final_state = self._graph.invoke(initial_state)

        chunks = final_state["retrieved_chunks"]
        result = AgenticResult(
            question               = question,
            answer                 = final_state["answer"],
            sub_questions          = final_state["sub_questions"],
            retrieved_chunks       = chunks,
            contexts               = [c.text for c in chunks],
            retrieval_attempts     = final_state["retrieval_attempts"],
            is_faithful            = final_state["is_faithful"],
            confidence             = final_state["confidence"],
            faithfulness_reasoning = final_state["faithfulness_reasoning"],
            question_type          = final_state["question_type"],
            model                  = self._model,
        )

        print(f"{'─'*65}")
        print(f"  Type          : {result.question_type}")
        print(f"  Sub-questions : {result.sub_questions}")
        print(f"  Chunks        : {len(result.retrieved_chunks)} ({result.retrieval_attempts} retries)")
        print(f"  Faithful      : {result.is_faithful} (confidence={result.confidence})")
        print(f"  Answer        : {result.answer[:150]}...")
        print(f"{'='*65}\n")

        return result

    def run_batch(self, questions: list[str]) -> list[AgenticResult]:
        """Run a list of questions and return results in order."""
        results = []
        for i, q in enumerate(questions, 1):
            print(f"\n[{i}/{len(questions)}]")
            results.append(self.run(q))
        return results


##############################################################################
#                              SMOKE TEST                                    #
##############################################################################

if __name__ == "__main__":
    pipeline = AgenticPipeline(model="qwen")

    test_questions = [
        # L1 — direct extraction
        "What was Apple's total net revenue for fiscal year 2024?",
        # L2 — cross-period comparison
        "How did Apple's gross margin change between Q3 2023 and Q3 2024?",
        # L3 — management commentary
        "What did Apple's management say about the outlook for Services revenue growth in the Q3 2024 earnings call?",
        # L4 — multi-source
        "Apple's 10-K disclosed a significant increase in R&D spending in FY2024 — did management address the rationale for this in the earnings call?",
    ]

    for q in test_questions:
        pipeline.run(q)
