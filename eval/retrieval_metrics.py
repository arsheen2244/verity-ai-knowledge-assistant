"""
Pure ranking metrics for retrieval -- no LLM, no ChromaDB, no network.

A question is labeled in eval_questions.json with ONE of:
  "relevant_pages": [{"source": "policy.pdf", "page": 3}, ...]
        -> a retrieved chunk is relevant if it comes from that file + page
  "must_contain":   ["3 days per week", ...]
        -> a retrieved chunk is relevant if its text contains that phrase
           (case-insensitive). Easier to label: copy a short phrase from the
           passage that answers the question.

Each label entry is one "target". Metrics at cutoff k:
  Hit@k     1 if any of the top-k chunks is relevant, else 0
  Recall@k  fraction of targets covered by at least one top-k chunk
  MRR       1 / rank of the first relevant chunk (0 if none retrieved)
"""
from __future__ import annotations

import os

Target = tuple  # ("page", source, page) | ("phrase", text)


def build_targets(item: dict) -> list[Target]:
    targets: list[Target] = []
    for p in item.get("relevant_pages", []) or []:
        targets.append(("page", os.path.basename(str(p["source"])).lower(), int(p["page"])))
    for phrase in item.get("must_contain", []) or []:
        phrase = str(phrase).strip().lower()
        if phrase:
            targets.append(("phrase", phrase))
    return targets


def target_matches(hit: dict, target: Target) -> bool:
    if target[0] == "page":
        return os.path.basename(str(hit.get("source", ""))).lower() == target[1] and int(hit.get("page", -1)) == target[2]
    return target[1] in str(hit.get("text", "")).lower()


def score_ranking(hits: list[dict], targets: list[Target], ks: list[int]) -> dict:
    """Scores one ranked hit list against one question's targets."""
    if not targets:
        raise ValueError("score_ranking needs at least one target")

    first_rank = None
    for rank, hit in enumerate(hits, 1):
        if any(target_matches(hit, t) for t in targets):
            first_rank = rank
            break

    result: dict = {"mrr": (1.0 / first_rank) if first_rank else 0.0}
    for k in ks:
        top = hits[:k]
        covered = sum(1 for t in targets if any(target_matches(h, t) for h in top))
        result[f"hit@{k}"] = 1.0 if any(any(target_matches(h, t) for t in targets) for h in top) else 0.0
        result[f"recall@{k}"] = covered / len(targets)
    return result


def aggregate(per_question: list[dict]) -> dict:
    """Mean of each metric across questions."""
    if not per_question:
        return {}
    keys = per_question[0].keys()
    return {k: sum(q[k] for q in per_question) / len(per_question) for k in keys}
