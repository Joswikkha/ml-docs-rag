"""
rag_chain.py — Phase 2c: LangChain QA Chain using FREE Groq LLM
Uses Groq's free API with Llama 3.1 70B — fast, free, excellent quality.

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
import logging
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()
log = logging.getLogger(__name__)

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
    Model: llama-3.1-70b-versatile — better than GPT-3.5, completely free.
    """
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise ValueError(
            "GROQ_API_KEY not found in .env file\n"
            "Get a free key at: https://console.groq.com/keys"
        )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
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

    log.info("RAG chain built — model: llama-3.1-70b-versatile (Groq)")
    return chain


# ── Ask Helper ─────────────────────────────────────────────────────
def ask(chain, question: str) -> dict:
    """
    Ask a question and get a structured response.

    Returns:
        {
            "answer"   : str   — the LLM answer
            "sources"  : list  — source URLs
            "contexts" : list  — raw retrieved chunk texts
        }
    """
    result = chain.invoke({"query": question})

    sources  = []
    contexts = []

    for doc in result.get("source_documents", []):
        src = doc.metadata.get("source", "Unknown source")
        if src not in sources:
            sources.append(src)
        contexts.append(doc.page_content)

    return {
        "answer"   : result["result"],
        "sources"  : sources,
        "contexts" : contexts,
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
    print("Testing RAG chain with Groq (Llama 3.1 70B) + hybrid")
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

    print("\n RAG chain test complete — ready for Phase 3 evaluation!")
