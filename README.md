# ML Documentation RAG System
> Chat with ML documentation. Benchmark 3 retrieval strategies. Evaluate with real RAGAS.

---

## What this project does

- Ingests **HuggingFace model cards** + **scikit-learn docs** into a vector store
- Supports **3 retrieval strategies**: dense, hybrid BM25+dense, hybrid+reranking
- Evaluates all 3 with **real RAGAS** (LLM-as-judge) — faithfulness, plus latency and token usage per call
- Serves a **Streamlit chat app** with example questions, retrieval-strategy explanations, and a benchmark dashboard

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/Joswikkha/ml-docs-rag.git
cd ml-docs-rag
pip install -r requirements.txt

# 2. Set your API keys
cp env.example .env
# Edit .env with your GROQ_API_KEY and COHERE_API_KEY (both free tier)

# 3. Ingest documents
python -m src.ingest

# 4. Build the vector store
python -m src.vectorstore

# 5. Run the RAGAS + latency/token benchmark
python -m src.run_benchmark --max-questions 10

# 6. Launch the app
streamlit run app.py
```

---

## Project structure

```
ml-docs-rag/
├── app.py                      # Streamlit entry point
├── pages/
│   ├── 1_QA_Chat.py             # chat UI, example questions, strategy explainer
│   └── 2_Benchmark_Dashboard.py # results dashboard
├── src/
│   ├── ingest.py                 # scrape + chunk HF model cards & sklearn docs
│   ├── vectorstore.py             # MiniLM embeddings → ChromaDB
│   ├── retrievers.py               # 3 retrieval strategies
│   ├── rag_chain.py                 # Groq LLM chain + latency/token tracking
│   ├── evaluate.py                   # real RAGAS scoring (LLM-as-judge)
│   └── run_benchmark.py               # runs all strategies, writes results CSV
├── data/
│   ├── processed/                # chunked docs ready for embedding
│   └── eval/eval_dataset.json    # 25 QA pairs for RAGAS
├── results/
│   └── benchmark_results.csv     # RAGAS + latency + token results per strategy
├── requirements.txt
├── env.example
└── README.md
```

---

## RAGAS Benchmark Results

Real results from `python -m src.run_benchmark --max-questions 10`, 10 questions per
strategy, run against the full-scale corpus (~240 HuggingFace model cards + 20
scikit-learn doc pages):

| Strategy                | Faithfulness | Avg Latency | Avg Tokens |
|--------------------------|:------------:|:-----------:|:----------:|
| Dense (baseline)         | 0.436        | 0.80s       | 924        |
| Hybrid + Reranking       | 0.515        | 1.71s       | 1,074      |
| Hybrid BM25+Dense ⭐     | **0.534**    | 1.64s       | 1,432      |

Both hybrid strategies clearly beat the dense baseline. Interestingly, plain
**Hybrid BM25+Dense edged out Hybrid+Reranking** on this run — the opposite order from
an earlier, smaller 3-question sample where reranking came out ahead. This is reported
as-is rather than smoothed over: with only 10 questions, the gap between the two hybrid
strategies (0.534 vs 0.515) is well within noise, and a larger eval set would be needed
to say confidently which one is actually better. It's also plausible that scaling the
corpus up (more, noisier candidate chunks for the reranker to sort through) shifts the
balance between the two differently than it did on the smaller corpus — worth digging
into further rather than assuming reranking always wins.

**Note on scope:** faithfulness is currently the only RAGAS metric scored by default
(`answer_relevancy`, `context_recall`, `context_precision` are implemented in
`src/evaluate.py` but disabled by default — see `METRICS` in that file). This was a
pragmatic call while developing against Groq's free-tier rate limits, where each
additional metric multiplies the number of LLM-judge calls per question. Re-enabling
them just means changing one line.

---

## Tech stack

| Layer       | Library / Model                              |
|-------------|-----------------------------------------------|
| LLM         | Groq — `openai/gpt-oss-120b` (free tier)      |
| Judge LLM   | Groq — `qwen/qwen3.8-27b` (separate quota from the generator LLM) |
| Embeddings  | `sentence-transformers/all-MiniLM-L6-v2` (local, free) |
| Vector DB   | ChromaDB (local)                              |
| Retrieval   | LangChain + `rank_bm25` + Cohere Rerank       |
| Evaluation  | `ragas` (real LLM-as-judge, not keyword overlap) |
| UI          | Streamlit                                     |

---

## Skills demonstrated

- End-to-end RAG pipeline design, entirely on free-tier APIs
- Multi-source document ingestion and metadata-aware chunking
- 3 retrieval strategy implementation and comparison (dense / hybrid / hybrid+rerank)
- Real quantitative evaluation with RAGAS (LLM-as-judge), plus latency and token
  instrumentation per call
- Handling real-world API rate limits: separating judge/generator model quotas,
  concurrency control, retry-with-backoff, and incremental result merging across runs
- Streamlit application development and deployment (Streamlit Community Cloud)