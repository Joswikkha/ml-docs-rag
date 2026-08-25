import streamlit as st

st.set_page_config(
    page_title="ML Docs RAG Benchmark",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 ML Documentation RAG System")
st.subheader("Retrieval Strategy Benchmarking")

st.markdown("""
This system benchmarks **3 retrieval strategies** on ML documentation
(Hugging Face model cards + Scikit-learn docs).

| # | Strategy | Description |
|---|----------|-------------|
| 1 | Dense (baseline) | Cosine similarity on embeddings |
| 2 | Hybrid BM25 + Dense | Keyword + semantic, merged with RRF |
| 3 | Hybrid + Reranking | Hybrid candidates reranked by Cohere |

### Tools used
`LangChain` · `ChromaDB` · `BM25` · `Cohere Rerank` · `Groq Llama 3.1` · `RAGAS` · `Streamlit`
""")

st.info("👈 Select a page from the sidebar to get started")

st.divider()
col1, col2, col3 = st.columns(3)
col1.metric("Documents indexed", "2,296 chunks")
col2.metric("Eval questions", "10 per strategy")
col3.metric("Strategies tested", "3")