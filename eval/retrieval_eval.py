"""
Retrieval-only evaluation: NO LLM, NO API key, runs fully offline on a laptop.

Compares the retrieval stages of this project against each other on your own
labeled questions, so the README can show real numbers for "hybrid + rerank
beats dense-only":

  dense          vector search only (ChromaDB)
  bm25           keyword search only
  hybrid         dense + BM25 merged with Reciprocal Rank Fusion
  hybrid+rerank  the above, re-scored by the cross-encoder  (the app's default)

Usage:
    1. Ingest your documents (UI or POST /api/ingest).
    2. Label questions in eval/eval_questions.json with "must_contain" or
       "relevant_pages" (see eval/retrieval_metrics.py for the format).
    3. python -m eval.retrieval_eval            # default cutoffs 1,3,5
       python -m eval.retrieval_eval --ks 1,4,8
    4. Read eval/retrieval_report.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.keyword_search import get_keyword_index  # noqa: E402
from app.reranker import _get_cross_encoder, rerank  # noqa: E402
from app.vector_store import get_store  # noqa: E402
from eval.retrieval_metrics import aggregate, build_targets, score_ranking  # noqa: E402

QUESTIONS_PATH = Path(__file__).parent / "eval_questions.json"
REPORT_PATH = Path(__file__).parent / "retrieval_report.md"
CONFIGS = ["dense", "bm25", "hybrid", "hybrid+rerank"]


def retrieve(config: str, query: str, kmax: int) -> list[dict]:
    """Runs ONE retrieval configuration and returns up to kmax ranked hits."""
    store = get_store()
    settings = get_settings()
    cand = max(settings.hybrid_candidate_k, kmax)

    if config == "dense":
        return store._dense_search(query, kmax)
    if config == "bm25":
        return get_keyword_index().search(store._collection, query, kmax)

    dense = store._dense_search(query, cand)
    keyword = get_keyword_index().search(store._collection, query, cand)
    fused = store._reciprocal_rank_fusion([dense, keyword], settings.rrf_k)
    if config == "hybrid":
        return fused[:kmax]
    if config == "hybrid+rerank":
        return rerank(query, fused, kmax)
    raise ValueError(config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval-only evaluation (no LLM needed)")
    parser.add_argument("--ks", default="1,3,5", help="comma-separated cutoffs, e.g. 1,3,5")
    args = parser.parse_args()
    ks = sorted({int(k) for k in args.ks.split(",") if k.strip()})
    kmax = max(ks)

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]
    labeled = [(q, build_targets(q)) for q in questions]
    labeled = [(q, t) for q, t in labeled if t and not q["question"].startswith("Example:")]
    skipped = len(questions) - len(labeled)
    if not labeled:
        raise SystemExit(
            "No usable labeled questions. Add real questions with 'must_contain' or "
            "'relevant_pages' to eval/eval_questions.json (Example: placeholders are ignored)."
        )

    store = get_store()
    if store.count() == 0:
        raise SystemExit("The vector store is empty -- ingest documents first (UI or POST /api/ingest).")

    reranker_loaded = _get_cross_encoder() is not None
    if not reranker_loaded:
        print("WARNING: cross-encoder could not be loaded, so 'hybrid+rerank' == 'hybrid' in this run.", file=sys.stderr)

    print(f"{len(labeled)} labeled questions ({skipped} skipped), {store.count()} chunks indexed.")
    per_config: dict[str, list[dict]] = {c: [] for c in CONFIGS}
    misses: list[str] = []
    for q, targets in labeled:
        for config in CONFIGS:
            hits = retrieve(config, q["question"], kmax)
            scores = score_ranking(hits, targets, ks)
            per_config[config].append(scores)
            if config == "hybrid+rerank" and scores[f"hit@{kmax}"] == 0.0:
                misses.append(q["question"])

    metric_keys = [f"hit@{k}" for k in ks] + [f"recall@{k}" for k in ks] + ["mrr"]
    lines = [
        "# Retrieval evaluation report (no LLM used)",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}Z",
        f"Labeled questions: {len(labeled)} (skipped {skipped} without labels)",
        f"Chunks indexed: {store.count()}",
        f"Candidates per retriever: {get_settings().hybrid_candidate_k}, RRF k: {get_settings().rrf_k}",
        f"Cross-encoder loaded: {reranker_loaded}",
        "",
        "| Config | " + " | ".join(metric_keys) + " |",
        "|---|" + "---|" * len(metric_keys),
    ]
    for config in CONFIGS:
        agg = aggregate(per_config[config])
        lines.append(f"| {config} | " + " | ".join(f"{agg[m]:.2f}" for m in metric_keys) + " |")

    lines += ["", f"## Questions where hybrid+rerank found nothing relevant in the top {kmax}", ""]
    lines += [f"- {m}" for m in misses] or ["- none"]
    lines += [
        "",
        "Hit@k: a relevant chunk appears in the top k. Recall@k: share of labeled targets "
        "covered. MRR: mean of 1/rank of the first relevant chunk. With a small question "
        "set these numbers are indicative, not statistically strong -- report the question "
        "count alongside them.",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {REPORT_PATH}\n")
    print("\n".join(lines[8 : 8 + 2 + len(CONFIGS)]))


if __name__ == "__main__":
    main()
