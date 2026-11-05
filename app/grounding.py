"""
Plain-code guardrails around the LLM's answer. No model, no network.

Two checks, both deterministic:

1. Citation check -- the prompt asks the model to cite "(source, page X)".
   A model can invent a page it never saw. verify_citations() parses every
   citation out of the answer and confirms each (source, page) pair is one
   of the chunks that were actually retrieved for this question.

2. Quote check -- any text the answer puts in double quotes must appear in
   the retrieved text (whitespace-insensitive, because PDF extraction
   inserts stray spaces and line breaks).

A refusal helper is included so the API can tell "the model declined" apart
from "the model answered".
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

REFUSAL_MESSAGE = "Not found in the documents."

# Matches (file.pdf, page 3) / (file.pdf, Page 3) / (file.pdf, p. 3).
_CITATION_RE = re.compile(
    r"\(\s*(?P<source>[^(),]+?)\s*,\s*(?:page|p\.?)\s*(?P<page>\d+)\s*\)",
    re.IGNORECASE,
)
# Double-quoted spans long enough to be a real quotation, not a scare quote.
_QUOTE_RE = re.compile(r"[\"“](?P<quote>[^\"”]{15,})[\"”]")
_REFUSAL_RE = re.compile(r"\bI don'?t know\b|\bnot found in the documents\b", re.IGNORECASE)


def _squash(text: str) -> str:
    return "".join(str(text).lower().split())


def _norm_source(name: str) -> str:
    return os.path.basename(str(name)).strip().lower()


@dataclass
class GroundingReport:
    citations_found: int = 0
    unverified_citations: list[str] = field(default_factory=list)
    quotes_found: int = 0
    unverified_quotes: list[str] = field(default_factory=list)

    @property
    def verified(self) -> bool:
        """True when nothing the answer cited or quoted is unsupported."""
        return not self.unverified_citations and not self.unverified_quotes

    def to_dict(self) -> dict:
        return {
            "verified": self.verified,
            "citations_found": self.citations_found,
            "unverified_citations": self.unverified_citations,
            "quotes_found": self.quotes_found,
            "unverified_quotes": self.unverified_quotes,
        }


def is_refusal(answer: str) -> bool:
    return bool(_REFUSAL_RE.search(answer or ""))


def verify_citations(answer: str, hits: list[dict]) -> GroundingReport:
    """Checks every citation and quotation in `answer` against `hits`
    (the retrieved chunks, each with source / page / text)."""
    report = GroundingReport()
    retrieved_pairs = {(_norm_source(h["source"]), int(h["page"])) for h in hits}
    retrieved_text = _squash(" ".join(h["text"] for h in hits))

    for m in _CITATION_RE.finditer(answer or ""):
        report.citations_found += 1
        pair = (_norm_source(m.group("source")), int(m.group("page")))
        if pair not in retrieved_pairs:
            report.unverified_citations.append(m.group(0))

    for m in _QUOTE_RE.finditer(answer or ""):
        report.quotes_found += 1
        if _squash(m.group("quote")) not in retrieved_text:
            report.unverified_quotes.append(m.group("quote"))

    return report


def below_threshold(hits: list[dict], min_rerank_score: float | None) -> bool:
    """True when retrieval is too weak to answer from, so the caller should
    refuse without calling the LLM.

    Uses the cross-encoder score of the best hit. If no threshold is
    configured, or the hits carry no rerank_score (reranker disabled or
    unavailable), this never refuses -- it fails open rather than blocking
    every question."""
    if min_rerank_score is None or not hits:
        return False
    scores = [h["rerank_score"] for h in hits if "rerank_score" in h]
    if not scores:
        return False
    return max(scores) < min_rerank_score
