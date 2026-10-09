"""Pure-Python tests for eval/retrieval_metrics.py (no models, no ChromaDB)."""
import pytest

from eval.retrieval_metrics import aggregate, build_targets, score_ranking


def _hit(text, source="a.pdf", page=1):
    return {"text": text, "source": source, "page": page}


def test_phrase_target_hit_and_mrr():
    targets = build_targets({"must_contain": ["3 days per week"]})
    hits = [_hit("unrelated"), _hit("Staff may work remotely 3 Days Per Week."), _hit("other")]
    s = score_ranking(hits, targets, [1, 3])
    assert s["hit@1"] == 0.0 and s["hit@3"] == 1.0
    assert s["mrr"] == pytest.approx(0.5)


def test_page_target_matches_source_and_page():
    targets = build_targets({"relevant_pages": [{"source": "Handbook.pdf", "page": 4}]})
    hits = [_hit("x", "handbook.pdf", 3), _hit("y", "handbook.pdf", 4)]
    s = score_ranking(hits, targets, [1, 2])
    assert s["hit@1"] == 0.0 and s["hit@2"] == 1.0 and s["recall@2"] == 1.0


def test_recall_counts_fraction_of_targets():
    targets = build_targets({"must_contain": ["alpha", "beta"]})
    s = score_ranking([_hit("alpha only")], targets, [1])
    assert s["recall@1"] == pytest.approx(0.5)


def test_no_relevant_hits_scores_zero():
    targets = build_targets({"must_contain": ["zzz"]})
    s = score_ranking([_hit("a"), _hit("b")], targets, [2])
    assert s == {"mrr": 0.0, "hit@2": 0.0, "recall@2": 0.0}


def test_aggregate_means():
    assert aggregate([{"mrr": 1.0}, {"mrr": 0.0}]) == {"mrr": 0.5}


def test_empty_targets_rejected():
    with pytest.raises(ValueError):
        score_ranking([_hit("a")], [], [1])
