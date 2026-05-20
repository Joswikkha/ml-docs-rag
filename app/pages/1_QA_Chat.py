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

st.set_page_config(page_title="Q&A Interface", page_icon="💬", layout="wide")
st.title("💬 Ask the ML Documentation")

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("⚙️ Settings")
    strategy = st.selectbox(
        "Retrieval Strategy",
        ["Strategy 3 — Hybrid + Rerank (Best)",
         "Strategy 2 — Hybrid BM25 + Dense",
         "Strategy 1 — Dense (Baseline)"],
        index=0
    )
    st.divider()
    st.header("📜 History")
    for i, item in enumerate(st.session_state.history[-5:]):
        st.caption(f"Q{i+1}: {item['question'][:35]}...")
    if st.button("Clear history"):
        st.session_state.history = []

@st.cache_resource
def load_chain(strategy_name):
    if "Rerank" in strategy_name:
        retriever = get_hybrid_rerank_retriever()
    elif "Hybrid" in strategy_name:
        retriever = get_hybrid_retriever()
    else:
        retriever = get_dense_retriever()
    return build_rag_chain(retriever=retriever)

query = st.text_input(
    "Your question",
    placeholder="e.g. What is the difference between BERT and DistilBERT?"
)
ask = st.button("Ask", type="primary")

if ask and query:
    with st.spinner(f"Retrieving with {strategy}..."):
        chain = load_chain(strategy)
        result = chain.invoke({"query": query})

    answer  = result.get("result", str(result))
    sources = result.get("source_documents", [])

    st.session_state.history.append({
        "question": query,
        "answer":   answer,
        "strategy": strategy
    })

    st.subheader("Answer")
    st.success(answer)

    if sources:
        st.subheader(f"Source Chunks ({len(sources)})")
        for i, doc in enumerate(sources):
            src = doc.metadata.get("source", "unknown")
            with st.expander(f"Chunk {i+1} · {src[:60]}"):
                 st.link_button("🔗 Open Source", src)
                 st.markdown( f"<div style='font-size:14px; line-height:1.6'>{doc.page_content[:400]}...</div>",unsafe_allow_html=True)
                

elif ask and not query:
    st.warning("Please type a question first.")