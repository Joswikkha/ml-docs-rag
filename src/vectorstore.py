"""
vectorstore.py — Phase 2a: Build and load ChromaDB vector store

Uses FREE HuggingFace embeddings — no API key, no credit card needed.
Model: all-MiniLM-L6-v2 (downloads automatically, ~90MB, runs locally)

Run ONCE to build the index:
    python src/vectorstore.py

After that, other scripts call load_vectorstore().
"""

import json
import logging
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()

# ── Logging ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.json"
PERSIST_DIR = str(ROOT / "data" / "chroma_db")

# ── Embedding model (free, runs locally) ──────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks() -> list[Document]:
    """Load chunks saved by ingest.py."""
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"chunks.json not found at {CHUNKS_PATH}\n"
            "Run Phase 1 first:  python src/ingest.py --hf-limit 10"
        )
    data = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    docs = [Document(page_content=d["text"], metadata=d["meta"]) for d in data]
    log.info(f"Loaded {len(docs)} chunks from {CHUNKS_PATH}")
    return docs


def get_embeddings():
    """
    Return FREE HuggingFace embedding model.
    Downloads model on first run (~90MB), then uses local cache.
    No API key needed.
    """
    log.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    log.info("First run downloads ~90MB — please wait...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    log.info("Embedding model loaded")
    return embeddings


def build_vectorstore(chunks: list[Document]) -> Chroma:
    """
    Embed all chunks and persist to ChromaDB.
    100% free — runs entirely on your local machine.
    Takes 3–8 minutes for 2296 chunks on a laptop CPU.
    """
    log.info(f"Building vector store from {len(chunks)} chunks...")
    log.info("Running locally — no API calls, completely free!")
    log.info("This takes 3–8 minutes on a laptop. Please wait...")

    embeddings = get_embeddings()

    BATCH_SIZE = 200
    all_docs   = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        all_docs.extend(batch)
        pct = int((min(i + BATCH_SIZE, len(chunks)) / len(chunks)) * 100)
        log.info(f"  {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)} chunks ready ({pct}%)...")

    log.info("Embedding and storing in ChromaDB — this is the slow part...")

    vectordb = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )

    count = vectordb._collection.count()
    log.info(f"Vector store built — {count} chunks indexed")
    log.info(f"Saved to {PERSIST_DIR}")
    return vectordb


def load_vectorstore() -> Chroma:
    """Load existing ChromaDB index from disk."""
    if not Path(PERSIST_DIR).exists():
        raise FileNotFoundError(
            f"ChromaDB not found at {PERSIST_DIR}\n"
            "Run this first:  python src/vectorstore.py"
        )
    embeddings = get_embeddings()
    vectordb   = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )
    count = vectordb._collection.count()
    log.info(f"Vector store loaded — {count} chunks available")
    return vectordb


if __name__ == "__main__":
    chunks = load_chunks()
    build_vectorstore(chunks)
    print("\nVector store ready — run retrievers.py next!")
