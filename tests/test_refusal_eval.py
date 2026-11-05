"""Tests the threshold sweep arithmetic in eval/refusal_eval.py (pure Python)."""
from eval.refusal_eval import candidate_thresholds, sweep


def test_sweep_counts_strictly_below_threshold():
    answerable = [3.0, 1.0, -2.0]
    unanswerable = [-6.0, -4.0, 0.5]
    rows = sweep(answerable, unanswerable, [-3.0])
    r = rows[0]
    assert r["correct_refusal"] == 2 / 3   # -6 and -4 are below -3
    assert r["false_refusal"] == 0.0       # nothing answerable is below -3


def test_sweep_trades_off_as_threshold_rises():
    answerable = [3.0, 1.0, -2.0]
    unanswerable = [-6.0, -4.0, 0.5]
    low, high = sweep(answerable, unanswerable, [-5.0, 2.0])
    assert high["correct_refusal"] >= low["correct_refusal"]
    assert high["false_refusal"] >= low["false_refusal"]


def test_perfect_separation_gives_perfect_balanced_accuracy():
    rows = sweep([2.0, 3.0], [-5.0, -4.0], [0.0])
    assert rows[0]["balanced_accuracy"] == 1.0


def test_candidate_thresholds_span_observed_range():
    ts = candidate_thresholds([-10.0, 5.0], steps=3)
    assert ts[0] == -10.0 and ts[-1] == 5.0 and len(ts) == 4
    assert candidate_thresholds([1.0, 1.0]) == [1.0]
