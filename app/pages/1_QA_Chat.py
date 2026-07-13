import streamlit as st
import sys
import os
from dotenv import load_dotenv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

load_dotenv(os.path.join(ROOT, ".env"))

from src.rag_chain import build_rag_chain
from src.retrievers import (
    get_dense_retriever,
    get_hybrid_retriever,
    get_hybrid_rerank_retriever
)

st.set_page_config(page_title="Q&A Chatbot", page_icon="💬", layout="wide")
st.title("💬 Chat with the ML Documentation")
st.caption(
    "Answers are grounded strictly in the indexed HuggingFace model cards "
    "and scikit-learn docs — the bot won't make things up outside that context."
)

# ── Conversation state ───────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {role, content, sources?, strategy?}

STRATEGY_INFO = {
    "Strategy 1 — Dense (Baseline)": {
        "short": "Pure semantic vector search.",
        "how": (
            "1. Your question is embedded into a vector using "
            "`all-MiniLM-L6-v2`.\n"
            "2. ChromaDB finds the **k=5** chunks whose embeddings are "
            "closest by cosine similarity.\n"
            "3. Those chunks are stuffed into the prompt and sent to the LLM.\n\n"
            "**Good at:** paraphrased / conceptual questions.\n"
            "**Weak at:** exact model names, parameter names, or rare terms "
            "that embeddings can blur together."
        ),
    },
    "Strategy 2 — Hybrid BM25 + Dense": {
        "short": "Keyword search + semantic search, merged.",
        "how": (
            "1. Runs **two retrievers in parallel** on the same question:\n"
            "   - `BM25Retriever` — classic keyword/term-frequency search "
            "(good for exact words).\n"
            "   - Dense retriever — the same embedding search as Strategy 1.\n"
            "2. Each retriever returns its own top-k ranked list.\n"
            "3. `EnsembleRetriever` merges both lists with **Reciprocal Rank "
            "Fusion (RRF)**, weighted 50/50, so a chunk ranked high in "
            "*either* list moves up.\n\n"
            "**Good at:** exact model names, hyperparameter names, rare "
            "technical terms that dense search alone can miss."
        ),
    },
    "Strategy 3 — Hybrid + Rerank (Best)": {
        "short": "Hybrid retrieval, then reranked by a cross-encoder.",
        "how": (
            "1. Runs the same Hybrid BM25+Dense step as Strategy 2, but "
            "casts a **wider net** — top 15 candidates instead of 5.\n"
            "2. Sends all 15 candidates + the question to **Cohere's "
            "`rerank-english-v3.0`** cross-encoder model.\n"
            "3. The reranker scores each chunk for how well it actually "
            "answers *this specific question* (not just similarity), and "
            "the **top 5** highest-scored chunks are kept.\n\n"
            "**Good at:** overall best precision — it corrects cases where "
            "hybrid retrieval ranked a mediocre chunk highly. Costs an "
            "extra API call, so it's slightly slower."
        ),
    },
}

EXAMPLE_QUESTIONS = [
    "— Pick an example question —",
    "What is the difference between BERT and DistilBERT?",
    "What does the n_estimators parameter do in RandomForest?",
    "What are the known limitations of GPT-2?",
    "How does a Random Forest handle overfitting?",
    "What is the difference between L1 and L2 regularization?",
    "What preprocessing does DistilBERT expect for input text?",
    "When should I use GradientBoosting over RandomForest?",
]

with st.sidebar:
    st.header("⚙️ Settings")
    strategy = st.selectbox(
        "Retrieval Strategy",
        ["Strategy 3 — Hybrid + Rerank (Best)",
         "Strategy 2 — Hybrid BM25 + Dense",
         "Strategy 1 — Dense (Baseline)"],
        index=0
    )
    with st.expander("ℹ️ How does this strategy work?", expanded=False):
        info = STRATEGY_INFO[strategy]
        st.caption(info["short"])
        st.markdown(info["how"])

    st.divider()
    st.header("💡 Example questions")
    st.caption("Pick one to ask it straight away.")
    example_choice = st.selectbox(
        "Examples",
        EXAMPLE_QUESTIONS,
        key="example_select",
        label_visibility="collapsed"
    )

    st.divider()
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()


@st.cache_resource
def load_chain(strategy_name):
    if "Rerank" in strategy_name:
        retriever = get_hybrid_rerank_retriever()
    elif "Hybrid" in strategy_name:
        retriever = get_hybrid_retriever()
    else:
        retriever = get_dense_retriever()
    return build_rag_chain(retriever=retriever)


def answer_question(question: str, strategy_name: str):
    """Runs the RAG chain and returns (answer, sources, error)."""
    try:
        chain = load_chain(strategy_name)
        result = chain.invoke({"query": question})
    except Exception as e:
        return None, [], str(e)

    answer  = result.get("result", str(result))
    sources = result.get("source_documents", [])
    return answer, sources, None


def render_sources(sources):
    if not sources:
        return
    with st.expander(f"📚 Source chunks ({len(sources)})"):
        for i, doc in enumerate(sources):
            src = doc.metadata.get("source", "unknown")
            st.markdown(f"**Chunk {i+1}** · `{src[:70]}`")
            if src and src != "unknown":
                st.link_button("🔗 Open source", src, key=f"src_{id(doc)}_{i}")
            st.markdown(
                f"<div style='font-size:14px; line-height:1.6'>"
                f"{doc.page_content[:400]}...</div>",
                unsafe_allow_html=True
            )
            st.divider()


def ask_and_store(question: str):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner(f"Retrieving with {strategy}..."):
            answer, sources, error = answer_question(question, strategy)

        if error:
            msg = (
                "Something went wrong while generating the answer. This is "
                "usually a missing API key (`GROQ_API_KEY` / `COHERE_API_KEY`) "
                "in the app's Secrets, not a problem with your question.\n\n"
                f"Details: `{error}`"
            )
            st.error(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
            return

        st.markdown(answer)
        render_sources(sources)
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "strategy": strategy,
        })


# ── Render existing conversation ─────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            render_sources(msg["sources"])

if not st.session_state.messages:
    st.info(
        "👋 Ask me anything about the indexed HuggingFace model cards or "
        "scikit-learn docs — or pick an example question from the sidebar."
    )

# ── Handle an example pick from the sidebar ──────────────────────────
if example_choice != EXAMPLE_QUESTIONS[0] and \
        st.session_state.get("_last_example") != example_choice:
    st.session_state["_last_example"] = example_choice
    ask_and_store(example_choice)

# ── Chat input at the bottom ─────────────────────────────────────────
user_question = st.chat_input("Ask a question about the ML documentation…")
if user_question:
    ask_and_store(user_question)