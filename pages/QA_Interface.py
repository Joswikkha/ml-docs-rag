import streamlit as st
import sys, os

# Fix the src import path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.rag_chain import get_rag_chain

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

strategy_map = {
    "Strategy 3 — Hybrid + Rerank (Best)": 3,
    "Strategy 2 — Hybrid BM25 + Dense":    2,
    "Strategy 1 — Dense (Baseline)":       1,
}
strategy_num = strategy_map[strategy]

@st.cache_resource
def load_chain(num):
    return get_rag_chain(strategy=num)

query = st.text_input(
    "Your question",
    placeholder="e.g. What is the difference between BERT and DistilBERT?"
)
ask = st.button("Ask", type="primary")

if ask and query:
    chain = load_chain(strategy_num)
    with st.spinner(f"Retrieving with {strategy}..."):
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
                st.write(doc.page_content[:400] + "...")

elif ask and not query:
    st.warning("Please type a question first.")