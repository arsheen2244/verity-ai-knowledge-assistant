"""
Shows WHY questions miss, so you fix the right thing.

For every labeled question where hybrid+rerank has no relevant chunk in the
top-k, it reports whether the labeled answer exists ANYWHERE in the index and
how deep any retriever ranked it:

  LABEL NOT FOUND IN INDEX   -> the label is wrong/too strict, or the PDF text
                                was extracted differently (fix the label)
  FOUND, rank > candidates   -> retrieval never surfaced it (fix chunking or
                                embeddings, raise hybrid_candidate_k)
  FOUND in candidates, but   -> the reranker pushed it out of top-k (try a
  not in top-k                  bigger top_k or a stronger reranker)

Usage (after ingesting):  python -m eval.diagnose_misses [--k 5]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.vector_store import get_store  # noqa: E402
from eval.retrieval_eval import QUESTIONS_PATH, retrieve  # noqa: E402
from eval.retrieval_metrics import build_targets, target_matches  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    store = get_store()
    everything = store._collection.get(include=["documents", "metadatas"])
    all_chunks = [
        {"text": t, **(m or {})} for t, m in zip(everything["documents"], everything["metadatas"])
    ]

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]
    for q in questions:
        targets = build_targets(q)
        if not targets or q["question"].startswith("Example:"):
            continue
        top = retrieve("hybrid+rerank", q["question"], args.k)
        if any(target_matches(h, t) for h in top for t in targets):
            continue

        print("=" * 78)
        print("MISS:", q["question"])
        print("labels:", [t[-1] for t in targets])
        holders = [c for c in all_chunks if any(target_matches(c, t) for t in targets)]
        if not holders:
            print("  -> LABEL NOT FOUND IN INDEX (label too strict / text extracted differently)")
        else:
            wide = retrieve("hybrid", q["question"], 50)
            ranks = [i for i, h in enumerate(wide, 1) if any(target_matches(h, t) for t in targets)]
            print(f"  -> answer exists in {len(holders)} chunk(s); best hybrid rank in top-50: {ranks[0] if ranks else 'not in top 50'}")
            print(f"     answer chunk starts: {holders[0]['text'][:160]!r}")
        print("  retrieved instead:")
        for h in top:
            print(f"     - p{h.get('page')}: {h['text'][:90]!r}")


if __name__ == "__main__":
    main()