"""
Refusal-threshold evaluation: NO LLM, runs offline once documents are ingested.

For every question in eval/eval_questions.json this takes the best
cross-encoder score among the top retrieved chunks, then sweeps a
MIN_RERANK_SCORE threshold and reports, for each value:

  correct refusals   share of UNANSWERABLE questions (no label) that would be
                     refused because retrieval scored below the threshold
  false refusals     share of ANSWERABLE questions (labeled) that would be
                     refused by mistake

A good threshold refuses the unanswerable questions without refusing real
ones. The two rates trade off, so the table is the result; the "suggested"
line is just the value with the best balanced accuracy.

Usage:
    1. Ingest your documents (UI or POST /api/ingest).
    2. python -m eval.refusal_eval
    3. Read eval/refusal_report.md, pick a threshold, set MIN_RERANK_SCORE in .env.

With few unanswerable questions (the shipped set has 5) the correct-refusal
rate moves in steps of 20 points, so add more unanswerable questions before
trusting a threshold.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

QUESTIONS_PATH = Path(__file__).parent / "eval_questions.json"
REPORT_PATH = Path(__file__).parent / "refusal_report.md"
TOP_K = 5


def sweep(answerable: list[float], unanswerable: list[float], thresholds: list[float]) -> list[dict]:
    """Pure function: for each threshold, the correct-refusal and false-refusal
    rates. A question is refused when its best score is strictly below the
    threshold, matching app.grounding.below_threshold."""
    rows = []
    for t in thresholds:
        correct = sum(1 for s in unanswerable if s < t) / len(unanswerable) if unanswerable else 0.0
        false = sum(1 for s in answerable if s < t) / len(answerable) if answerable else 0.0
        rows.append(
            {
                "threshold": t,
                "correct_refusal": correct,
                "false_refusal": false,
                "balanced_accuracy": (correct + (1.0 - false)) / 2.0,
            }
        )
    return rows


def candidate_thresholds(scores: list[float], steps: int = 15) -> list[float]:
    """Evenly spaced thresholds across the observed score range."""
    lo, hi = min(scores), max(scores)
    if lo == hi:
        return [lo]
    return [round(lo + (hi - lo) * i / steps, 2) for i in range(steps + 1)]


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app.config import get_settings
    from app.reranker import _get_cross_encoder
    from app.vector_store import get_store
    from eval.retrieval_eval import retrieve
    from eval.retrieval_metrics import build_targets

    if _get_cross_encoder() is None:
        raise SystemExit("The cross-encoder could not be loaded, so there are no rerank scores to evaluate.")
    store = get_store()
    if store.count() == 0:
        raise SystemExit("The vector store is empty -- ingest documents first (UI or POST /api/ingest).")

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]
    answerable: list[float] = []
    unanswerable: list[float] = []
    for q in questions:
        if q["question"].startswith("Example:"):
            continue
        hits = retrieve("hybrid+rerank", q["question"], TOP_K)
        scores = [h["rerank_score"] for h in hits if "rerank_score" in h]
        if not scores:
            continue
        best = max(scores)
        (answerable if build_targets(q) else unanswerable).append(best)

    if not answerable or not unanswerable:
        raise SystemExit("Need both labeled (answerable) and unlabeled (unanswerable) questions.")

    rows = sweep(answerable, unanswerable, candidate_thresholds(answerable + unanswerable))
    best_row = max(rows, key=lambda r: r["balanced_accuracy"])

    lines = [
        "# Refusal threshold report (no LLM used)",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}Z",
        f"Answerable questions: {len(answerable)}; unanswerable questions: {len(unanswerable)}",
        f"Chunks indexed: {store.count()}; best score taken over the top {TOP_K} chunks",
        f"Cross-encoder: {get_settings().rerank_model_name}",
        "",
        "| MIN_RERANK_SCORE | Correct refusals | False refusals | Balanced accuracy |",
        "|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['threshold']} | {r['correct_refusal']:.2f} | {r['false_refusal']:.2f} | {r['balanced_accuracy']:.2f} |"
        )
    lines += [
        "",
        f"Suggested: MIN_RERANK_SCORE={best_row['threshold']} "
        f"(correct refusals {best_row['correct_refusal']:.2f}, false refusals {best_row['false_refusal']:.2f}).",
        "",
        f"Only {len(unanswerable)} unanswerable questions: each one is worth "
        f"{100 / len(unanswerable):.0f} points of correct-refusal rate, so treat this as indicative.",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}\n")
    print("\n".join(lines[8:]))


if __name__ == "__main__":
    main()
