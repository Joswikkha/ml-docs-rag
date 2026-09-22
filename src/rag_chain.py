"""
rag_chain.py — Phase 2c: LangChain QA Chain using FREE Groq LLM
Uses Groq's free API with GPT-OSS 120B — fast, free, excellent quality.

Usage:
    from rag_chain import build_rag_chain, ask
    from retrievers import get_retriever

    retriever = get_retriever("hybrid_rerank")
    chain     = build_rag_chain(retriever)
    result    = ask(chain, "What is BERT?")
    print(result["answer"])
    print(result["sources"])
"""

import os
import time
import logging
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.callbacks import BaseCallbackHandler

load_dotenv()
log = logging.getLogger(__name__)


# ── Metrics callback: latency + token usage ────────────────────────
class MetricsCallback(BaseCallbackHandler):
    """
    Captures, per chain.invoke() call:
      - retrieval_time_s : wall-clock time spent in the top-level retriever
                           (parent_run_id is None -> ignores nested sub-
                           retriever calls inside EnsembleRetriever /
                           ContextualCompressionRetriever)
      - llm_time_s       : wall-clock time spent in the LLM generation call
      - prompt_tokens / completion_tokens / total_tokens : from Groq's
        OpenAI-compatible usage stats on the LLM response
    """
    def __init__(self):
        self._retrieval_t0 = None
        self._llm_t0 = None
        self.retrieval_time_s = None
        self.llm_time_s = None
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def on_retriever_start(self, serialized, query, *, run_id, parent_run_id=None, **kwargs):
        if parent_run_id is None:
            self._retrieval_t0 = time.perf_counter()

    def on_retriever_end(self, documents, *, run_id, parent_run_id=None, **kwargs):
        if parent_run_id is None and self._retrieval_t0 is not None:
            self.retrieval_time_s = round(time.perf_counter() - self._retrieval_t0, 3)

    def on_llm_start(self, serialized, prompts, **kwargs):
        self._llm_t0 = time.perf_counter()

    def on_llm_end(self, response, **kwargs):
        if self._llm_t0 is not None:
            self.llm_time_s = round(time.perf_counter() - self._llm_t0, 3)
        try:
            usage = (response.llm_output or {}).get("token_usage", {})
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)
            self.total_tokens += usage.get("total_tokens", 0)
        except Exception:
            pass

# ── Prompt Template ────────────────────────────────────────────────
PROMPT_TEMPLATE = """
You are a helpful ML documentation assistant.
Answer the question using ONLY the context provided below.
If the answer is not in the context, say exactly:
"I don't have that information in my documentation."

Do NOT make up information. Do NOT use your own training knowledge.
Always end your answer with a "Source:" line citing where you found the answer.

Context:
{context}

Question: {question}

Answer (end with Source: <document name>):
"""


# ── Build Chain ────────────────────────────────────────────────────
def build_rag_chain(retriever, temperature: float = 0):
    """
    Build a RetrievalQA chain using FREE Groq LLM.
    Model: openai/gpt-oss-120b — Groq's current recommended replacement for the deprecated Llama 3.3 70B, completely free.
    """
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise ValueError(
            "GROQ_API_KEY not found in .env file\n"
            "Get a free key at: https://console.groq.com/keys"
        )

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=temperature,
        groq_api_key=groq_key
    )

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    log.info("RAG chain built — model: openai/gpt-oss-120b (Groq)")
    return chain


# ── Ask Helper ─────────────────────────────────────────────────────
def ask(chain, question: str) -> dict:
    """
    Ask a question and get a structured response, including
    performance metrics for that single call.

    Returns:
        {
            "answer"            : str   — the LLM answer
            "sources"           : list  — source URLs
            "contexts"          : list  — raw retrieved chunk texts
            "total_time_s"      : float — full chain.invoke() wall time
            "retrieval_time_s"  : float | None — time spent retrieving
            "llm_time_s"        : float | None — time spent generating
            "prompt_tokens"     : int
            "completion_tokens" : int
            "total_tokens"      : int
        }
    """
    metrics = MetricsCallback()

    t0 = time.perf_counter()
    result = chain.invoke({"query": question}, config={"callbacks": [metrics]})
    total_time_s = round(time.perf_counter() - t0, 3)

    sources  = []
    contexts = []

    for doc in result.get("source_documents", []):
        src = doc.metadata.get("source", "Unknown source")
        if src not in sources:
            sources.append(src)
        contexts.append(doc.page_content)

    return {
        "answer"            : result["result"],
        "sources"           : sources,
        "contexts"          : contexts,
        "total_time_s"      : total_time_s,
        "retrieval_time_s"  : metrics.retrieval_time_s,
        "llm_time_s"        : metrics.llm_time_s,
        "prompt_tokens"     : metrics.prompt_tokens,
        "completion_tokens" : metrics.completion_tokens,
        "total_tokens"      : metrics.total_tokens,
    }


# ── Quick test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    from retrievers import get_retriever

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )

    TEST_QUESTIONS = [
        "What is the difference between BERT and DistilBERT?",
        "What does the n_estimators parameter do in RandomForest?",
        "What are the known limitations of GPT-2?",
    ]

    print("\n" + "=" * 60)
    print("Testing RAG chain with Groq (GPT-OSS 120B) + hybrid")
    print("=" * 60)

    retriever = get_retriever("hybrid", k=5)
    chain     = build_rag_chain(retriever)

    for q in TEST_QUESTIONS:
        print(f"\n Question: {q}")
        print("-" * 50)
        result = ask(chain, q)
        print(f" Answer: {result['answer']}")
        print(f" Sources:")
        for s in result["sources"]:
            print(f"   - {s}")
        print(f" ⏱  Total: {result['total_time_s']}s "
              f"(retrieval: {result['retrieval_time_s']}s, "
              f"llm: {result['llm_time_s']}s)")
        print(f" 🔢 Tokens: {result['prompt_tokens']} prompt + "
              f"{result['completion_tokens']} completion = "
              f"{result['total_tokens']} total")

    print("\n RAG chain test complete — ready for Phase 3 evaluation!")