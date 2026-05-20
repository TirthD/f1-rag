"""Evaluation runner — Stage 1, visual grading.

Runs every question in eval/questions.json through the pipeline, prints
expected vs actual side by side for quick visual grading, and saves the
full results to eval/results/run_<timestamp>.json so we can compare runs
later (Stage 1 baseline vs Stage 2 changes).

Usage:
    python -m eval.run_eval

Optional:
    python -m eval.run_eval --quiet     # skip retrieved-chunks display
    python -m eval.run_eval --limit 3   # only run first 3 questions (for testing)
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from src.pipeline import build_chain
from src.config import PROJECT_ROOT

EVAL_DIR = PROJECT_ROOT / "eval"
QUESTIONS_PATH = EVAL_DIR / "questions.json"
RESULTS_DIR = EVAL_DIR / "results"


def load_questions() -> dict:
    with open(QUESTIONS_PATH) as f:
        return json.load(f)


def truncate(text: str, n: int = 80) -> str:
    """Shorten a string for terminal display."""
    text = text.replace("\n", " ").strip()
    return text if len(text) <= n else text[: n - 3] + "..."


def run_one(chain, retriever, question: str, show_chunks: bool):
    """Run a single question, return (answer, retrieved_chunks, elapsed_seconds)."""
    start = time.time()
    retrieved = retriever.invoke(question)
    answer = chain.invoke(question)
    elapsed = time.time() - start

    chunks_summary = [
        {
            "race": d.metadata.get("race", "?"),
            "snippet": truncate(d.page_content, 150),
        }
        for d in retrieved
    ]
    return answer, chunks_summary, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true",
                    help="Don't print retrieved chunks (faster scan)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only run the first N questions")
    args = ap.parse_args()

    data = load_questions()
    questions = data["questions"]
    if args.limit:
        questions = questions[: args.limit]

    print("=" * 80)
    print(f"F1 RAG Eval Run — {datetime.now().isoformat(timespec='seconds')}")
    print(f"Running {len(questions)} questions")
    print("=" * 80)

    # Build the chain ONCE, not per-question. Saves loading the embedding
    # model and vector store on every iteration.
    print("\nLoading pipeline...")
    chain, retriever = build_chain()
    print("Ready.\n")

    results = []
    for i, q in enumerate(questions, 1):
        qid = q["id"]
        question = q["question"]
        expected = q["expected_answer"]
        category = q["category"]

        print("\n" + "─" * 80)
        print(f"[{i}/{len(questions)}] {qid}  ({category})")
        print(f"Q: {question}")
        print(f"Expected: {truncate(expected, 200)}")

        try:
            answer, chunks, elapsed = run_one(chain, retriever, question, not args.quiet)
            status = "ok"
        except Exception as e:
            answer = f"[ERROR: {e}]"
            chunks = []
            elapsed = 0.0
            status = "error"

        if not args.quiet and chunks:
            print("Retrieved:")
            for j, c in enumerate(chunks, 1):
                print(f"  [{j}] ({c['race']}) {c['snippet']}")

        print(f"\nActual ({elapsed:.1f}s):")
        # Print full answer indented so it stands out
        for line in answer.split("\n"):
            print(f"  {line}")

        results.append({
            "id": qid,
            "category": category,
            "question": question,
            "expected_answer": expected,
            "actual_answer": answer,
            "retrieved_chunks": chunks,
            "elapsed_seconds": round(elapsed, 2),
            "status": status,
            "notes": q.get("notes", ""),
            # Filled in by human grading after the run
            "grade": None,
            "grader_note": None,
        })

    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"run_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "stage": "stage_1_naive",
            "results": results,
        }, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Done. {len(results)} questions run.")
    print(f"Total time: {sum(r['elapsed_seconds'] for r in results):.1f}s")
    print(f"Results saved to: {out_path.relative_to(PROJECT_ROOT)}")
    print("=" * 80)
    print("\nNext: review the output above and grade each question.")
    print("You can edit the 'grade' field in the JSON to 'correct', 'partial',")
    print("'wrong', or 'refused' for tracking, but it's optional for Stage 1.")


if __name__ == "__main__":
    main()