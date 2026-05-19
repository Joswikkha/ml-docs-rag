"""
evaluate.py — Phase 3a: Local evaluation scoring (no OpenAI needed)
Scores a RAG chain using 4 metrics calculated locally:

  - faithfulness       : keyword overlap between answer and retrieved context
  - answer_relevancy   : keyword overlap between answer and question
  - context_recall     : overlap between retrieved context and ground truth
  - context_precision  : how focused/relevant the retrieved chunks are

100% free — no API keys needed for scoring.

Usage:
    from evaluate import run_local_eval
    scores = run_local_eval(qa_pairs, chain, "Dense (baseline)")
"""

import re
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

ROOT      = Path(__file__).resolve().parent.parent
EVAL_PATH = ROOT / "data" / "eval" / "eval_dataset.json"


def load_eval_dataset() -> list[dict]:
    """Load the 25 QA pairs from eval_dataset.json"""
    if not EVAL_PATH.exists():
        raise FileNotFoundError(f"eval_dataset.json not found at {EVAL_PATH}")
    pairs = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    log.info(f"Loaded {len(pairs)} evaluation QA pairs")
    return pairs


# ── Local scoring helpers ──────────────────────────────────────────

def _tokenize(text: str) -> set:
    """Lowercase + split into word tokens, remove stopwords."""
    stopwords = {
        "a","an","the","is","it","in","of","to","and","or","for",
        "with","on","at","by","this","that","are","was","be","as",
        "from","has","have","had","not","but","what","which","who",
        "how","when","where","does","do","did","can","will","its",
        "their","they","we","you","i","me","my","your","our","been"
    }
    words = re.findall(r'\b[a-z][a-z0-9]*\b', text.lower())
    return set(w for w in words if w not in stopwords and len(w) > 2)


def _overlap_score(text_a: str, text_b: str) -> float:
    """
    Jaccard-style overlap between two texts.
    Score = shared words / total unique words
    Range: 0.0 (no overlap) to 1.0 (identical)
    """
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union        = tokens_a | tokens_b
    return round(len(intersection) / len(union), 4)


def score_single(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str,
) -> dict:
    """
    Score one QA result on 4 local metrics.

    faithfulness      : answer grounded in retrieved context?
    answer_relevancy  : answer addresses the question?
    context_recall    : context covers the ground truth?
    context_precision : context is focused (not noisy)?
    """
    full_context = " ".join(contexts)

    # Faithfulness: how much of the answer is in the context
    faithfulness = _overlap_score(answer, full_context)

    # Answer relevancy: how well answer addresses question
    answer_relevancy = _overlap_score(answer, question)

    # Context recall: how much of ground truth is in the context
    context_recall = _overlap_score(full_context, ground_truth)

    # Context precision: how relevant are chunks to the question
    if contexts:
        chunk_scores = [_overlap_score(c, question) for c in contexts]
        context_precision = round(sum(chunk_scores) / len(chunk_scores), 4)
    else:
        context_precision = 0.0

    return {
        "faithfulness"      : faithfulness,
        "answer_relevancy"  : answer_relevancy,
        "context_recall"    : context_recall,
        "context_precision" : context_precision,
    }


# ── Main eval runner ───────────────────────────────────────────────

def run_local_eval(
    qa_pairs: list[dict],
    chain,
    strategy_name: str,
    max_questions: int = 10,
) -> dict:
    """
    Run local evaluation on a RAG chain.

    Args:
        qa_pairs      : list of {"question": str, "ground_truth": str}
        chain         : built RAG chain from rag_chain.py
        strategy_name : label (e.g. "Dense (baseline)")
        max_questions : how many questions to score

    Returns:
        dict with strategy name and 4 mean scores
    """
    log.info(f"Running local eval — strategy: {strategy_name}")
    log.info(f"Evaluating {max_questions}/{len(qa_pairs)} questions...")

    pairs   = qa_pairs[:max_questions]
    all_scores = {
        "faithfulness"      : [],
        "answer_relevancy"  : [],
        "context_recall"    : [],
        "context_precision" : [],
    }
    success = 0

    for i, pair in enumerate(pairs, 1):
        question     = pair["question"]
        ground_truth = pair["ground_truth"]

        log.info(f"  [{i}/{max_questions}] {question[:65]}...")

        try:
            result   = chain.invoke({"query": question})
            answer   = result["result"]
            contexts = [d.page_content for d in result.get("source_documents", [])]

            scores = score_single(question, answer, contexts, ground_truth)

            for metric, val in scores.items():
                all_scores[metric].append(val)

            success += 1

        except Exception as e:
            log.warning(f"  Skipped question {i}: {e}")
            continue

    if success == 0:
        raise RuntimeError("No questions were evaluated successfully")

    # Mean scores
    summary = {
        "strategy"          : strategy_name,
        "faithfulness"      : round(sum(all_scores["faithfulness"]) / success, 4),
        "answer_relevancy"  : round(sum(all_scores["answer_relevancy"]) / success, 4),
        "context_recall"    : round(sum(all_scores["context_recall"]) / success, 4),
        "context_precision" : round(sum(all_scores["context_precision"]) / success, 4),
        "questions_scored"  : success,
    }

    log.info(f"Results for {strategy_name}:")
    log.info(f"  Faithfulness      : {summary['faithfulness']}")
    log.info(f"  Answer Relevancy  : {summary['answer_relevancy']}")
    log.info(f"  Context Recall    : {summary['context_recall']}")
    log.info(f"  Context Precision : {summary['context_precision']}")
    log.info(f"  Questions scored  : {success}/{max_questions}")

    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    pairs = load_eval_dataset()
    print(f"Eval dataset loaded: {len(pairs)} questions")
    print(f"Sample: {pairs[0]['question']}")
    print("Local scoring ready — no API keys needed!")
