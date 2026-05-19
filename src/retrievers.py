"""
retrievers.py — Phase 2b: 3 retrieval strategies

Strategy 1: Dense vector search (baseline)
Strategy 2: Hybrid BM25 + Dense (better for rare terms)
Strategy 3: Hybrid + Cohere Reranking (best quality)

Usage:
    from retrievers import get_retriever
    retriever = get_retriever("hybrid_rerank", chunks)
    docs = retriever.invoke("What is BERT?")
"""

import os
import json
import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_cohere import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
from dotenv import load_dotenv

from src.vectorstore import load_vectorstore

load_dotenv()

log = logging.getLogger(__name__)
# ── Paths ──────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.json"


def load_chunks_for_bm25() -> list[Document]:
    """Load chunks from disk for BM25 (needs raw text)."""
    data = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    return [Document(page_content=d["text"], metadata=d["meta"]) for d in data]


# ══════════════════════════════════════════════════════════════════
# Strategy 1 — Dense Vector Search (Baseline)
# ══════════════════════════════════════════════════════════════════
def get_dense_retriever(k: int = 5):
    """
    Pure semantic vector search.
    Fast and simple — your baseline to beat.
    Weakness: misses rare terms like exact model names.
    """
    log.info("Loading Strategy 1: Dense retriever")
    vectordb  = load_vectorstore()
    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    log.info("✓ Strategy 1 ready")
    return retriever


# ══════════════════════════════════════════════════════════════════
# Strategy 2 — Hybrid BM25 + Dense (Reciprocal Rank Fusion)
# ══════════════════════════════════════════════════════════════════
def get_hybrid_retriever(k: int = 5):
    """
    Combines BM25 keyword search + dense vector search.
    Results merged with Reciprocal Rank Fusion (RRF).
    Better for: exact model names, parameter names, rare terms.
    """
    log.info("Loading Strategy 2: Hybrid BM25 + Dense retriever")

    # Dense side
    vectordb    = load_vectorstore()
    dense_ret   = vectordb.as_retriever(search_kwargs={"k": k})

    # BM25 side
    chunks      = load_chunks_for_bm25()
    bm25_ret    = BM25Retriever.from_documents(chunks)
    bm25_ret.k  = k

    # Merge with equal weights
    hybrid = EnsembleRetriever(
        retrievers=[bm25_ret, dense_ret],
        weights=[0.5, 0.5]   # tune: 0.4/0.6 favours dense
    )
    log.info("✓ Strategy 2 ready")
    return hybrid


# ══════════════════════════════════════════════════════════════════
# Strategy 3 — Hybrid + Cohere Reranking (Best Quality)
# ══════════════════════════════════════════════════════════════════
def get_hybrid_rerank_retriever(k: int = 5, rerank_top_n: int = 15):
    """
    Hybrid retrieval → Cohere reranker picks best k from top rerank_top_n.
    Most accurate strategy — the one you'll present as your winner.

    How it works:
      1. Hybrid retriever fetches top 15 candidates
      2. Cohere reranker scores all 15 by relevance to the query
      3. Returns top 5 highest-scored chunks
    """
    log.info("Loading Strategy 3: Hybrid + Cohere Rerank retriever")

    # Get hybrid retriever with wider net (rerank_top_n instead of k)
    base_retriever = get_hybrid_retriever(k=rerank_top_n)

    # Cohere reranker
    cohere_key = os.getenv("COHERE_API_KEY")
    if not cohere_key:
        raise ValueError("COHERE_API_KEY not found in .env file")

    compressor = CohereRerank(
        cohere_api_key=cohere_key,
        top_n=k,
        model="rerank-english-v3.0"
    )

    reranked = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    log.info("✓ Strategy 3 ready")
    return reranked


# ══════════════════════════════════════════════════════════════════
# Unified getter — use this everywhere
# ══════════════════════════════════════════════════════════════════
STRATEGY_NAMES = {
    "dense"         : "Dense (baseline)",
    "hybrid"        : "Hybrid BM25+Dense",
    "hybrid_rerank" : "Hybrid + Reranking",
}

def get_retriever(strategy: str = "dense", k: int = 5):
    """
    Get a retriever by strategy name.

    Args:
        strategy: one of "dense", "hybrid", "hybrid_rerank"
        k: number of chunks to return

    Returns:
        A LangChain retriever object
    """
    if strategy == "dense":
        return get_dense_retriever(k=k)
    elif strategy == "hybrid":
        return get_hybrid_retriever(k=k)
    elif strategy == "hybrid_rerank":
        return get_hybrid_rerank_retriever(k=k)
    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            f"Choose from: {list(STRATEGY_NAMES.keys())}"
        )


# ── Quick test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    TEST_QUERY = "What is the difference between BERT and DistilBERT?"

    print("\n" + "═" * 55)
    print("Testing all 3 retrieval strategies")
    print("═" * 55)

    for strategy, label in STRATEGY_NAMES.items():
        print(f"\n▶ {label}")
        try:
            retriever = get_retriever(strategy, k=3)
            docs      = retriever.invoke(TEST_QUERY)
            print(f"  Retrieved {len(docs)} chunks")
            for i, d in enumerate(docs, 1):
                src = d.metadata.get("source", "unknown")
                print(f"  [{i}] {src[:70]}")
                print(f"      {d.page_content[:100].strip()}...")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    print("\n✅ Retriever test complete — ready for rag_chain.py!")
