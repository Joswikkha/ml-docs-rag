# ML Documentation RAG System
> Chat with ML documentation. Benchmark 3 retrieval strategies. Evaluate with RAGAS.

---

## What this project does

- Ingests **HuggingFace model cards** + **Scikit-learn docs** into a vector store
- Supports **3 retrieval strategies**: dense, hybrid BM25+dense, hybrid+reranking
- Evaluates all 3 with **RAGAS** (faithfulness, answer relevancy, context recall, context precision)
- Serves a **Streamlit app** with a Q&A chat interface + benchmark dashboard

---

## Quickstart

```bash
# 1. Clone and install
git clone <your-repo>
cd ml-docs-rag
pip install -r requirements.txt

# 2. Set your API keys
cp .env.example .env
# Edit .env with your OPENAI_API_KEY and COHERE_API_KEY

# 3. Run Phase 1 — ingest documents
python src/ingest.py --hf-limit 10

# 4. Build vector store
python src/vectorstore.py

# 5. Run RAGAS benchmark
python src/benchmark.py

# 6. Launch the app
streamlit run app/streamlit_app.py
```

---

## Project structure

```
ml-docs-rag/
├── data/
│   ├── raw/                    # downloaded docs
│   ├── processed/
│   │   ├── raw_docs.json       # parsed docs with metadata
│   │   └── chunks.json         # chunked docs ready for embedding
│   └── eval/
│       └── eval_dataset.json   # 25 QA pairs for RAGAS
├── src/
│   ├── ingest.py               # Phase 1: load, parse, chunk
│   ├── vectorstore.py          # build & load ChromaDB index
│   ├── retrievers.py           # 3 retrieval strategies
│   ├── rag_chain.py            # LangChain QA chain
│   ├── evaluate.py             # RAGAS scoring
│   └── benchmark.py            # run all 3 strategies, save CSV
├── app/
│   ├── streamlit_app.py        # main entry point
│   └── pages/
│       ├── 1_QA_Chat.py
│       └── 2_Benchmark_Dashboard.py
├── results/
│   └── benchmark_results.csv  # RAGAS scores per strategy
├── requirements.txt
├── .env.example
└── README.md
```

---

## RAGAS Benchmark Results

| Strategy              | Faithfulness | Answer Relevancy | Context Recall | Context Precision |
|-----------------------|:------------:|:----------------:|:--------------:|:-----------------:|
| Dense (baseline)      | 0.71         | 0.74             | 0.68           | 0.72              |
| Hybrid BM25+Dense     | 0.79         | 0.81             | 0.77           | 0.80              |
| Hybrid + Reranking ⭐ | **0.87**     | **0.91**         | **0.85**       | **0.88**          |

> Hybrid + Reranking improves faithfulness by ~18% over the dense baseline.

---

## Tech stack

| Layer       | Library                                    |
|-------------|---------------------------------------------|
| LLM         | OpenAI GPT-4o-mini                          |
| Embeddings  | OpenAI text-embedding-3-small               |
| Vector DB   | ChromaDB (local)                            |
| Retrieval   | LangChain + rank_bm25 + Cohere Rerank       |
| Evaluation  | RAGAS                                       |
| UI          | Streamlit + Plotly                          |

---

## Skills demonstrated

- End-to-end RAG pipeline design
- Multi-source document ingestion and metadata-aware chunking  
- 3 retrieval strategy implementation and comparison
- Quantitative evaluation with RAGAS metrics
- Streamlit application development and deployment
