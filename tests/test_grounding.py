"""
Tests for the plain-code guardrails in app/grounding.py. Pure Python: no
embeddings, no ChromaDB, no network.
"""
from app.grounding import REFUSAL_MESSAGE, below_threshold, is_refusal, verify_citations

HITS = [
    {
        "source": "policy.pdf",
        "page": 3,
        "text": "Employees may work remotely up to 3 days per week.",
    },
    {
        "source": "policy.pdf",
        "page": 4,
        "text": "Home office equipment up to $500 per year is reimbursable.",
    },
]


def test_real_citation_is_verified():
    report = verify_citations("Up to 3 days a week (policy.pdf, page 3).", HITS)
    assert report.citations_found == 1
    assert report.verified


def test_invented_page_is_flagged():
    report = verify_citations("Up to 3 days a week (policy.pdf, page 9).", HITS)
    assert report.unverified_citations == ["(policy.pdf, page 9)"]
    assert not report.verified


def test_invented_source_is_flagged():
    report = verify_citations("See (handbook.pdf, page 3).", HITS)
    assert not report.verified


def test_citation_format_variants_and_path_prefix():
    answer = "A (policy.pdf, Page 3) and B (/tmp/x/policy.pdf, p. 4)."
    report = verify_citations(answer, HITS)
    assert report.citations_found == 2
    assert report.verified


def test_quote_present_in_context_is_verified_despite_whitespace():
    answer = 'It says "Employees may work   remotely up to 3 days per week" (policy.pdf, page 3).'
    assert verify_citations(answer, HITS).verified


def test_fabricated_quote_is_flagged():
    answer = 'It says "Employees may work remotely every day of the week" (policy.pdf, page 3).'
    report = verify_citations(answer, HITS)
    assert report.quotes_found == 1
    assert len(report.unverified_quotes) == 1
    assert not report.verified


def test_short_quoted_terms_are_not_treated_as_quotations():
    report = verify_citations('The "Home" policy applies (policy.pdf, page 3).', HITS)
    assert report.quotes_found == 0
    assert report.verified


def test_answer_without_citations_is_trivially_verified():
    report = verify_citations("I don't know based on the available documents.", HITS)
    assert report.citations_found == 0
    assert report.verified


def test_is_refusal():
    assert is_refusal("I don't know based on the available documents.")
    assert is_refusal(REFUSAL_MESSAGE)
    assert not is_refusal("Up to 3 days per week (policy.pdf, page 3).")


def test_below_threshold_refuses_only_when_best_score_is_too_low():
    weak = [{"rerank_score": -8.0}, {"rerank_score": -5.5}]
    strong = [{"rerank_score": -8.0}, {"rerank_score": 2.1}]
    assert below_threshold(weak, -4.0)
    assert not below_threshold(strong, -4.0)


def test_below_threshold_fails_open():
    assert not below_threshold([{"rerank_score": -99.0}], None)  # disabled
    assert not below_threshold([], -4.0)  # no hits: handled elsewhere
    assert not below_threshold([{"text": "no score here"}], -4.0)  # reranker off
