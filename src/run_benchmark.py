"""
run_benchmark.py — Phase 3b: Full benchmark runner

Ties together retrievers.py + rag_chain.py + evaluate.py to produce a
single results/benchmark_results.csv covering all 3 retrieval strategies:

  - 4 RAGAS metrics (real LLM-as-judge scoring, not keyword overlap)
  - Total / retrieval / LLM latency per question
  - Prompt / completion / total token usage per question

Usage:
    python -m src.run_benchmark --max-questions 10
    python -m src.run_benchmark --max-questions 5 --strategies dense hybrid
"""

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
from groq import RateLimitError

from src.retrievers import get_retriever, STRATEGY_NAMES
from src.rag_chain import build_rag_chain, ask
from src.evaluate import load_eval_dataset, run_ragas_eval

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT        = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Seconds to wait between each question — keeps us comfortably under
# Groq's free-tier tokens-per-minute (TPM) limit, since RAGAS's own
# scoring calls just before this may have already used most of the
# current minute's budget.
PACING_DELAY_S = 5


def _ask_with_retry(chain, question: str, max_retries: int = 3):
    """Wraps ask() with a backoff retry for Groq 429 rate-limit errors."""
    for attempt in range(1, max_retries + 1):
        try:
            return ask(chain, question)
        except RateLimitError as e:
            wait_s = 15 * attempt
            log.warning(f"  Rate limited (attempt {attempt}/{max_retries}) — "
                        f"waiting {wait_s}s before retrying: {e}")
            time.sleep(wait_s)
    # last attempt — let it raise if it still fails
    return ask(chain, question)


def collect_perf_metrics(chain, questions: list[str]) -> pd.DataFrame:
    """Re-runs each question through ask() purely to capture latency and
    token usage per call (kept separate from the RAGAS run so RAGAS's own
    LLM-judge calls never get mixed into these numbers)."""
    rows = []
    for i, q in enumerate(questions, 1):
        log.info(f"  [{i}/{len(questions)}] measuring perf: {q[:60]}...")
        r = _ask_with_retry(chain, q)
        rows.append({
            "question"          : q,
            "total_time_s"      : r["total_time_s"],
            "retrieval_time_s"  : r["retrieval_time_s"],
            "llm_time_s"        : r["llm_time_s"],
            "prompt_tokens"     : r["prompt_tokens"],
            "completion_tokens" : r["completion_tokens"],
            "total_tokens"      : r["total_tokens"],
        })
        if i < len(questions):
            time.sleep(PACING_DELAY_S)
    return pd.DataFrame(rows)


def run_full_benchmark(strategies: list[str], max_questions: int) -> pd.DataFrame:
    qa_pairs = load_eval_dataset()
    all_rows = []

    for strategy_key in strategies:
        label = STRATEGY_NAMES[strategy_key]
        log.info("=" * 60)
        log.info(f"Strategy: {label}")
        log.info("=" * 60)

        retriever = get_retriever(strategy_key, k=5)
        chain     = build_rag_chain(retriever)

        # 1. Real RAGAS scores (LLM-as-judge)
        summary, ragas_df = run_ragas_eval(qa_pairs, chain, label, max_questions)

        log.info("  Cooling down before latency/token measurement "
                 f"({PACING_DELAY_S * 2}s)...")
        time.sleep(PACING_DELAY_S * 2)

        # 2. Latency + token usage per question
        perf_df = collect_perf_metrics(chain, [p["question"] for p in qa_pairs[:max_questions]])

        merged = ragas_df.reset_index(drop=True).join(
            perf_df.drop(columns=["question"]).reset_index(drop=True)
        )
        merged["strategy"] = label
        all_rows.append(merged)

        log.info(f"  Avg total time  : {perf_df['total_time_s'].mean():.2f}s")
        log.info(f"  Avg total tokens: {perf_df['total_tokens'].mean():.0f}")

        if strategy_key != strategies[-1]:
            log.info(f"  Cooling down before next strategy ({PACING_DELAY_S * 2}s)...")
            time.sleep(PACING_DELAY_S * 2)

    new_df   = pd.concat(all_rows, ignore_index=True)
    out_path = RESULTS_DIR / "benchmark_results.csv"

    # Merge with any existing results instead of overwriting — lets you
    # run one strategy today, another tomorrow, without losing earlier
    # runs. Any strategy you re-run replaces its OLD rows (not duplicated).
    if out_path.exists():
        existing_df = pd.read_csv(out_path)
        ran_labels  = set(new_df["strategy"].unique())
        kept_old    = existing_df[~existing_df["strategy"].isin(ran_labels)]
        full_df     = pd.concat([kept_old, new_df], ignore_index=True)
        log.info(f"  Merged with existing results — kept "
                 f"{len(kept_old)} old rows from other strategies")
    else:
        full_df = new_df

    full_df.to_csv(out_path, index=False)
    log.info(f"\n✓ Saved full benchmark → {out_path} ({len(full_df)} total rows)")
    return full_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full RAG benchmark (RAGAS + perf)")
    parser.add_argument(
        "--strategies", nargs="+",
        default=["dense", "hybrid", "hybrid_rerank"],
        choices=["dense", "hybrid", "hybrid_rerank"],
        help="Which strategies to benchmark (default: all 3)",
    )
    parser.add_argument(
        "--max-questions", type=int, default=10,
        help="How many eval questions to use per strategy (default: 10). "
             "Each question triggers several LLM-judge calls for RAGAS, "
             "so keep this modest on a free-tier API key.",
    )
    args = parser.parse_args()

    df = run_full_benchmark(args.strategies, args.max_questions)
    print("\n🎉 Benchmark complete —", len(df), "rows written to results/benchmark_results.csv")