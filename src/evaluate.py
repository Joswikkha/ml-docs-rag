"""
evaluate.py — Phase 3a: REAL RAGAS evaluation (LLM-as-judge under the hood)

Scores a RAG chain using the actual `ragas` library, not a hand-rolled
keyword-overlap approximation:

  - faithfulness       : does the answer's claims actually follow from
                          the retrieved context? (checked by an LLM judge
                          that breaks the answer into individual claims)
  - answer_relevancy    : does the answer actually address the question?
                          (an LLM generates candidate questions from the
                          answer, then compares their embeddings to the
                          original question's embedding)
  - context_recall      : does the retrieved context cover everything in
                          the ground-truth answer? (LLM judge)
  - context_precision    : are the retrieved chunks ranked with the most
                          relevant ones first? (LLM judge)

Uses your existing free stack — Groq LLM as the judge, HuggingFace
MiniLM as the embedder — so no OpenAI key is required.

Usage:
    from evaluate import run_ragas_eval
    summary = run_ragas_eval(qa_pairs, chain, "Dense (baseline)")
"""

import os
import json
import logging
from pathlib import Path

from datasets import Dataset
from dotenv import load_dotenv

from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()
log = logging.getLogger(__name__)

ROOT      = Path(__file__).resolve().parent.parent
EVAL_PATH = ROOT / "data" / "eval" / "eval_dataset.json"

# Full RAGAS metric set — all 4 pillars of RAG evaluation.
ALL_METRICS  = [faithfulness, answer_relevancy, context_recall, context_precision]

# Reduced set for use while the free-tier Groq quota is under heavy
# contention — faithfulness (is the answer grounded in context?) and
# answer_relevancy (does it address the question?) are the two most
# commonly cited RAGAS metrics in interviews, and together they're
# roughly half the LLM-judge call volume of the full 4-metric set.
LIGHT_METRICS = [faithfulness, answer_relevancy]

# Even lighter — just faithfulness. Across tonight's runs on the free
# tier, faithfulness has consistently returned real scores while
# answer_relevancy keeps timing out (it needs more sub-calls per
# question: generating several candidate questions from the answer,
# then embedding them — more chances to hit a 429 mid-way through).
# Use this when even LIGHT_METRICS is too much for current API load.
MINIMAL_METRICS = [faithfulness]

# Change this to LIGHT_METRICS or ALL_METRICS once your Groq quota has
# more headroom (e.g. on the Developer tier, or a quieter time of day).
METRICS = MINIMAL_METRICS


def load_eval_dataset() -> list[dict]:
    """Load the QA pairs from eval_dataset.json"""
    if not EVAL_PATH.exists():
        raise FileNotFoundError(f"eval_dataset.json not found at {EVAL_PATH}")
    pairs = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    log.info(f"Loaded {len(pairs)} evaluation QA pairs")
    return pairs


def _get_ragas_judge():
    """
    RAGAS needs an LLM (to judge claims) and an embedding model (for
    answer_relevancy's cosine-similarity check).

    Deliberately uses a DIFFERENT Groq model than the main RAG chain
    (which uses openai/gpt-oss-120b). Groq enforces rate limits
    per-model, not just per-organization — so a shared model would
    compete for the same token budget and hit 429s constantly.
    qwen/qwen3.8-27b draws from its own separate quota AND, being a
    stronger model than gpt-oss-20b, follows RAGAS's structured-output
    prompts (claim extraction, question generation, attribution
    classification) more reliably — gpt-oss-20b was silently failing
    those and producing 0.0/nan scores instead of real evaluation.
    """
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise ValueError(
            "GROQ_API_KEY not found in .env file — RAGAS needs an LLM "
            "judge to score faithfulness/relevancy/recall/precision."
        )

    judge_llm = ChatGroq(
        model="qwen/qwen3.8-27b",
        temperature=0,
        groq_api_key=groq_key,
    )
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    return LangchainLLMWrapper(judge_llm), LangchainEmbeddingsWrapper(embeddings)


def run_ragas_eval(
    qa_pairs: list[dict],
    chain,
    strategy_name: str,
    max_questions: int = 10,
):
    """
    Run REAL RAGAS evaluation on a RAG chain.

    Args:
        qa_pairs      : list of {"question": str, "ground_truth": str}
        chain         : built RAG chain from rag_chain.py
        strategy_name : label (e.g. "Dense (baseline)")
        max_questions : how many questions to score (each one costs
                        several LLM-judge calls — keep this modest on
                        a free-tier API key to avoid rate limits)

    Returns:
        (summary_dict, per_question_dataframe)
    """
    log.info(f"Running RAGAS eval — strategy: {strategy_name}")
    pairs = qa_pairs[:max_questions]

    questions, answers, contexts_list, ground_truths = [], [], [], []
    success = 0

    for i, pair in enumerate(pairs, 1):
        question     = pair["question"]
        ground_truth = pair["ground_truth"]
        log.info(f"  [{i}/{len(pairs)}] generating answer: {question[:60]}...")
        try:
            result   = chain.invoke({"query": question})
            answer   = result["result"]
            contexts = [d.page_content for d in result.get("source_documents", [])]
        except Exception as e:
            log.warning(f"  Skipped question {i}: {e}")
            continue

        questions.append(question)
        answers.append(answer)
        contexts_list.append(contexts)
        ground_truths.append(ground_truth)
        success += 1

    if success == 0:
        raise RuntimeError("No questions were answered successfully — nothing to score")

    dataset = Dataset.from_dict({
        "question"    : questions,
        "answer"      : answers,
        "contexts"    : contexts_list,
        "ground_truth": ground_truths,
    })

    log.info(f"Scoring {success} answers with RAGAS (LLM-as-judge)...")
    judge_llm, judge_embeddings = _get_ragas_judge()

    # Groq's free tier caps tokens-per-minute (TPM) fairly low (e.g. 8000
    # for openai/gpt-oss-120b). RAGAS defaults to firing up to 16 LLM
    # calls concurrently, which blows past that limit almost instantly.
    # max_workers=1 makes it fully sequential; max_wait/max_retries give
    # it room to back off and retry instead of hard-failing on a 429.
    safe_run_config = RunConfig(
        max_workers=1,
        max_wait=120,
        max_retries=15,
    )

    ragas_result = evaluate(
        dataset,
        metrics=METRICS,
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=safe_run_config,
    )

    scores_df = ragas_result.to_pandas()

    metric_cols = [m.name for m in METRICS]  # only columns that actually ran

    summary = {"strategy": strategy_name}
    for col in metric_cols:
        summary[col] = round(float(scores_df[col].mean()), 4)
    summary["questions_scored"] = success

    log.info(f"RAGAS results for {strategy_name}:")
    for col in metric_cols:
        log.info(f"  {col:<18}: {summary[col]}")
    log.info(f"  questions_scored : {success}/{len(pairs)}")

    return summary, scores_df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    pairs = load_eval_dataset()
    print(f"Eval dataset loaded: {len(pairs)} questions")
    print(f"Sample: {pairs[0]['question']}")
    print("Real RAGAS scoring ready — this will call your Groq + HF "
          "embeddings, no OpenAI key needed.")