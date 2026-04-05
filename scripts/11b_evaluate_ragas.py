"""
scripts/11b_evaluate_ragas.py
MediCare Pharmaceuticals India — Step 2 of 2 for RAGAS evaluation
------------------------------------------------------------------
Loads pre-collected answers from data/ragas/ragas_answers.json
and runs RAGAS metrics using local Llama-3 8B as critic.

Run AFTER 11a_collect_answers.py has finished.

Usage:
    python scripts/11b_evaluate_ragas.py                  # evaluate + compare vs baseline
    python scripts/11b_evaluate_ragas.py --save-baseline  # save as baseline (run once first)

Dependencies:
    pip install ragas datasets langchain-community
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime

try:
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas import evaluate
    from datasets import Dataset
except ImportError as e:
    sys.exit(f"[ERROR] Missing dependency: {e}\nRun: pip install ragas datasets")

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
ANSWERS_PATH  = PROJECT_ROOT / "data" / "ragas" / "ragas_answers.json"
BASELINE_PATH = PROJECT_ROOT / "data" / "ragas" / "ragas_baseline.json"
CSV_PATH      = PROJECT_ROOT / "data" / "ragas" / "ragas_results_latest.csv"


# ──────────────────────────────────────────────────────────────────────────
# LOCAL LLAMA CRITIC + METRICS
# Fresh Llama instance — no shared state with the agent process.
# ──────────────────────────────────────────────────────────────────────────
def configure_metrics_with_local_llm():
    print("[RAGAS] Loading local Llama-3 8B as critic LLM...")

    from langchain_community.llms import LlamaCpp
    from langchain_community.embeddings import HuggingFaceEmbeddings

    llama = LlamaCpp(
        model_path=str(PROJECT_ROOT / "models/llama-3-8b_300.Q4_K_M.gguf"),
        n_ctx=1024,        # reduced to avoid KV cache pressure
        max_tokens=800,    # DO NOT raise above 800
        temperature=0.1,
        n_gpu_layers=0,
        verbose=False,
    )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    critic_llm        = LangchainLLMWrapper(llama)
    critic_embeddings = LangchainEmbeddingsWrapper(embeddings)

    # Instantiate metrics with LLM and embeddings
    faithfulness = Faithfulness(llm=critic_llm)
    context_recall = ContextRecall(llm=critic_llm)
    answer_relevancy = AnswerRelevancy(llm=critic_llm, embeddings=critic_embeddings)

    print("[RAGAS] Critic ready.\n")

    return [faithfulness, answer_relevancy, context_recall]


# ──────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="RAGAS evaluation — MediCare")
    parser.add_argument("--save-baseline", action="store_true",
                        help="Save current scores as baseline for delta comparison")
    args = parser.parse_args()

    # Load pre-collected answers
    if not ANSWERS_PATH.exists():
        sys.exit(
            f"[ERROR] Answers file not found: {ANSWERS_PATH}\n"
            "Run 11a_collect_answers.py first."
        )

    with open(ANSWERS_PATH) as f:
        records = json.load(f)

    print(f"Loaded {len(records)} records from {ANSWERS_PATH}\n")

    # Configure critic and instantiate metrics
    metrics = configure_metrics_with_local_llm()

    # Build RAGAS dataset
    dataset = Dataset.from_dict({
        "question":     [r["question"]     for r in records],
        "answer":       [r["answer"]       for r in records],
        "contexts":     [r["contexts"]     for r in records],
        "ground_truth": [r["ground_truth"] for r in records],
    })

    # Run evaluation — batch_size=1 to prevent any parallel KV conflicts
    print("Running RAGAS evaluation (batch_size=1, ~3-5 min)...\n")
    t0 = time.time()
    result = evaluate(
        dataset,
        metrics=metrics,
        raise_exceptions=False,
        batch_size=1,
    )
    elapsed = time.time() - t0

    scores = {
        "faithfulness":     round(float(result["faithfulness"]), 4),
        "answer_relevancy": round(float(result["answer_relevancy"]), 4),
        "context_recall":   round(float(result["context_recall"]), 4),
        "timestamp":        datetime.now().isoformat(),
        "n_questions":      len(records),
        "eval_time_s":      round(elapsed, 1),
    }

    # Print summary
    print(f"\n{'='*60}")
    print("  RAGAS SCORES — MediCare Pharmaceuticals India")
    print(f"{'='*60}")
    print(f"  Faithfulness     : {scores['faithfulness']:.4f}  (target > 0.85)")
    print(f"  Answer Relevancy : {scores['answer_relevancy']:.4f}  (target > 0.80)")
    print(f"  Context Recall   : {scores['context_recall']:.4f}  (target > 0.75)")
    print(f"{'='*60}")
    print(f"  Evaluated at     : {scores['timestamp']}")
    print(f"  Critic LLM       : Local Llama-3 8B (CPU)")
    print(f"  Eval time        : {scores['eval_time_s']}s")
    print(f"  Questions        : {scores['n_questions']}")
    print(f"{'='*60}\n")

    # Per-question CSV
    try:
        df   = result.to_pandas()
        cols = [c for c in ["question","faithfulness","answer_relevancy","context_recall"] if c in df.columns]
        print("Per-question breakdown:")
        print(df[cols].to_string(index=False, max_colwidth=55))
        print()
        CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        df[cols].to_csv(CSV_PATH, index=False)
        print(f"Per-question CSV saved to: {CSV_PATH}\n")
    except Exception as e:
        print(f"[WARN] Per-question breakdown failed: {e}")

    # Save or compare baseline
    if args.save_baseline:
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(BASELINE_PATH, "w") as f:
            json.dump(scores, f, indent=2)
        print(f"Baseline saved to: {BASELINE_PATH}")
    elif BASELINE_PATH.exists():
        with open(BASELINE_PATH) as f:
            baseline = json.load(f)
        print("Delta vs saved baseline:")
        for metric in ["faithfulness", "answer_relevancy", "context_recall"]:
            delta = scores[metric] - baseline.get(metric, 0)
            arrow = "▲" if delta >= 0 else "▼"
            print(f"  {metric:<20}: {arrow} {delta:+.4f}")
        print()
    else:
        print("Tip: run with --save-baseline to save these scores for future comparisons.")


if __name__ == "__main__":
    main()
