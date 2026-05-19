"""
ingest.py — Phase 1: Load, parse, and chunk ML documentation
Sources:
  1. HuggingFace model cards (via HF Hub API)
  2. Scikit-learn HTML docs (via requests + BeautifulSoup)

Run:
    python src/ingest.py
Output:
    data/processed/chunks.json   ← chunked docs with metadata
    data/processed/raw_docs.json ← raw docs before chunking
"""


import os
import json
import time
import logging
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup
from huggingface_hub import HfApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ── Logging setup ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
PROCESSED   = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# 1. HuggingFace Model Cards
# ══════════════════════════════════════════════════════════════════

# Tasks to pull model cards for — edit this list to expand coverage
HF_TASKS = [
    "text-classification",
    "token-classification",
    "text-generation",
    "question-answering",
    "summarization",
    "fill-mask",
]

def fetch_hf_model_cards(
    tasks: list[str] = HF_TASKS,
    limit_per_task: int = 10,
    sleep_between: float = 0.3,
) -> list[Document]:
    """
    Pull README.md (model card) for top models in each HF task.
    Returns a list of LangChain Documents with rich metadata.
    """
    api  = HfApi()
    docs = []

    for task in tasks:
        log.info(f"Fetching HF model cards — task: {task}")
        try:
            models = list(api.list_models(
                filter=task,
                sort="downloads",
                direction=-1,
                limit=limit_per_task,
            ))
        except Exception as e:
            log.warning(f"  Failed to list models for {task}: {e}")
            continue

        for m in models:
            readme_url = (
                f"https://huggingface.co/{m.modelId}/raw/main/README.md"
            )
            try:
                resp = requests.get(readme_url, timeout=10)
                if resp.status_code != 200:
                    continue
                text = resp.text.strip()
                if len(text) < 100:          # skip near-empty cards
                    continue

                docs.append(Document(
                    page_content=text,
                    metadata={
                        "source"    : f"https://huggingface.co/{m.modelId}",
                        "doc_type"  : "hf_model_card",
                        "model_name": m.modelId,
                        "task_type" : task,
                        "downloads" : getattr(m, "downloads", 0),
                    }
                ))
                log.info(f"  ✓ {m.modelId} ({len(text):,} chars)")
                time.sleep(sleep_between)    # be polite to the API

            except Exception as e:
                log.warning(f"  Skipped {m.modelId}: {e}")
                continue

    log.info(f"HuggingFace: {len(docs)} model cards fetched")
    return docs


# ══════════════════════════════════════════════════════════════════
# 2. Scikit-learn Documentation
# ══════════════════════════════════════════════════════════════════

SKLEARN_PAGES = {
    "ensemble"    : "https://scikit-learn.org/stable/modules/ensemble.html",
    "svm"         : "https://scikit-learn.org/stable/modules/svm.html",
    "tree"        : "https://scikit-learn.org/stable/modules/tree.html",
    "linear_model": "https://scikit-learn.org/stable/modules/linear_model.html",
    "clustering"  : "https://scikit-learn.org/stable/modules/clustering.html",
    "neural_net"  : "https://scikit-learn.org/stable/modules/neural_networks_supervised.html",
    "preprocessing": "https://scikit-learn.org/stable/modules/preprocessing.html",
    "model_eval"  : "https://scikit-learn.org/stable/modules/model_evaluation.html",
}

def _scrape_sklearn_page(name: str, url: str) -> Optional[Document]:
    """Scrape a single sklearn doc page → clean text Document."""
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        log.warning(f"  Failed to fetch {url}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove noisy elements
    for tag in soup.select("nav, footer, .headerlink, .sphinxsidebar, "
                           ".related, script, style, .toctree-wrapper"):
        tag.decompose()

    # Extract main content only
    main = soup.find("div", {"class": "body"}) or soup.find("article") or soup
    text = main.get_text(separator="\n", strip=True)

    # Basic cleanup — collapse excessive blank lines
    lines   = text.splitlines()
    cleaned = "\n".join(
        line for i, line in enumerate(lines)
        if line.strip() or (i > 0 and lines[i-1].strip())
    )

    if len(cleaned) < 200:
        log.warning(f"  {name}: content too short, skipping")
        return None

    return Document(
        page_content=cleaned,
        metadata={
            "source"  : url,
            "doc_type": "sklearn_docs",
            "section" : name,
        }
    )


def fetch_sklearn_docs(pages: dict = SKLEARN_PAGES) -> list[Document]:
    """Scrape all configured sklearn doc pages."""
    docs = []
    for name, url in pages.items():
        log.info(f"Scraping sklearn: {name}")
        doc = _scrape_sklearn_page(name, url)
        if doc:
            docs.append(doc)
            log.info(f"  ✓ {name} ({len(doc.page_content):,} chars)")
        time.sleep(0.2)
    log.info(f"Scikit-learn: {len(docs)} pages fetched")
    return docs


# ══════════════════════════════════════════════════════════════════
# 3. Chunking
# ══════════════════════════════════════════════════════════════════

def chunk_documents(
    docs: list[Document],
    chunk_size: int  = 500,
    chunk_overlap: int = 100,
) -> list[Document]:
    """
    Split documents into overlapping chunks.
    Metadata is preserved on every chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    log.info(f"Chunking: {len(docs)} docs → {len(chunks)} chunks "
             f"(size={chunk_size}, overlap={chunk_overlap})")
    return chunks


# ══════════════════════════════════════════════════════════════════
# 4. Save helpers
# ══════════════════════════════════════════════════════════════════

def save_docs(docs: list[Document], path: Path) -> None:
    data = [{"text": d.page_content, "meta": d.metadata} for d in docs]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False),
    encoding="utf-8"
)
    log.info(f"Saved {len(docs)} items → {path}")


def load_chunks(path: Path) -> list[Document]:
    data = json.loads(path.read_text())
    return [Document(page_content=d["text"], metadata=d["meta"]) for d in data]


# ══════════════════════════════════════════════════════════════════
# 5. Main runner
# ══════════════════════════════════════════════════════════════════

def run_ingestion(
    hf_limit: int = 10,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[Document]:
    """
    Full ingestion pipeline.
    Returns chunked documents ready for embedding.
    """
    log.info("═" * 50)
    log.info("Phase 1 — ML Documentation Ingestion")
    log.info("═" * 50)

    # --- Fetch ---
    hf_docs  = fetch_hf_model_cards(limit_per_task=hf_limit)
    skl_docs = fetch_sklearn_docs()
    all_docs = hf_docs + skl_docs

    if not all_docs:
        raise RuntimeError("No documents fetched — check network / API access")

    # --- Save raw ---
    save_docs(all_docs, PROCESSED / "raw_docs.json")

    # --- Chunk ---
    chunks = chunk_documents(all_docs, chunk_size, chunk_overlap)

    # --- Save chunks ---
    save_docs(chunks, PROCESSED / "chunks.json")

    # --- Summary ---
    log.info("═" * 50)
    log.info("✓ Ingestion complete")
    log.info(f"  HF model cards : {len(hf_docs)}")
    log.info(f"  Sklearn pages  : {len(skl_docs)}")
    log.info(f"  Total raw docs : {len(all_docs)}")
    log.info(f"  Total chunks   : {len(chunks)}")
    log.info(f"  Saved to       : {PROCESSED}")
    log.info("═" * 50)

    return chunks


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ML Docs RAG — Phase 1 Ingestion")
    parser.add_argument("--hf-limit",       type=int, default=10,
                        help="Model cards per HF task (default: 10)")
    parser.add_argument("--chunk-size",     type=int, default=500,
                        help="Chunk size in characters (default: 500)")
    parser.add_argument("--chunk-overlap",  type=int, default=100,
                        help="Chunk overlap in characters (default: 100)")
    args = parser.parse_args()

    chunks = run_ingestion(
        hf_limit=args.hf_limit,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(f"\n🎉 Ready for Phase 2 — {len(chunks)} chunks indexed")
