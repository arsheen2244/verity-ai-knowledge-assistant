"""
Cross-encoder reranking.

Why this is a separate stage from retrieval: a bi-encoder (the embedding
model in embeddings.py) scores a question and a chunk independently, then
compares the two vectors -- fast enough to run over an entire corpus, but it
never lets the question and the chunk "see" each other while scoring. A
cross-encoder feeds the (question, chunk) pair into one model together, so
it can pick up on interactions a bi-encoder misses (negation, exact-phrase
overlap, which entity a pronoun refers to). That precision comes at a real
cost -- roughly O(n) full model passes instead of O(1) vector comparisons --
which is exactly why it only runs on the small post-hybrid-search shortlist
in vector_store.py, never on the whole store.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from .config import get_settings

logger = logging.getLogger(__name__)


@lru_cache
def _get_cross_encoder():
    """Returns a loaded CrossEncoder, or None if it can't be loaded (offline
    environment, no network to huggingface.co). Mirrors the same
    fail-soft pattern as embeddings.py's sentence-transformer loader."""
    try:
        from sentence_transformers import CrossEncoder

        settings = get_settings()
        return CrossEncoder(settings.rerank_model_name)
    except Exception as e:  # noqa: BLE001 -- deliberately broad: any failure -> skip reranking
        logger.warning(
            "Reranking disabled: could not load cross-encoder '%s' (%s). "
            "Falling back to the pre-rerank (RRF-fused) order.",
            get_settings().rerank_model_name,
            e,
        )
        return None


def rerank(query: str, hits: list[dict], top_k: int) -> list[dict]:
    """Re-scores `hits` (each a dict with a 'text' key) against `query` with
    a cross-encoder and returns the top_k, each carrying a `rerank_score`.
    If the model can't be loaded, returns the first top_k of `hits`
    unchanged so the caller never has to special-case unavailability."""
    if not hits:
        return hits

    model = _get_cross_encoder()
    if model is None:
        return hits[:top_k]

    pairs = [(query, h["text"]) for h in hits]
    scores = model.predict(pairs)
    for h, s in zip(hits, scores):
        h["rerank_score"] = float(s)
    return sorted(hits, key=lambda h: h["rerank_score"], reverse=True)[:top_k]
