"""
Glues retrieval (vector_store) and generation (llm) together, then runs the
plain-code guardrails from grounding.py on the result. Kept deliberately
thin -- if this file starts accumulating parsing or prompt logic, that's a
sign it belongs in document_processor.py, llm.py or grounding.py instead.
"""
from __future__ import annotations

from .config import get_settings
from .grounding import REFUSAL_MESSAGE, below_threshold, is_refusal, verify_citations
from .llm import generate_answer
from .vector_store import get_store


def _dedupe_sources(hits: list[dict]) -> list[dict]:
    seen = set()
    sources = []
    for h in hits:
        key = (h["source"], h["page"])
        if key not in seen:
            seen.add(key)
            sources.append({"source": h["source"], "page": h["page"]})
    return sources


def answer_question(question: str, top_k: int | None = None) -> dict:
    store = get_store()
    hits = store.search(question, top_k=top_k)

    # Weak retrieval -> refuse in code, before the LLM can improvise.
    if below_threshold(hits, get_settings().min_rerank_score):
        return {
            "answer": REFUSAL_MESSAGE,
            "sources": [],
            "retrieved_chunks": len(hits),
            "refused": True,
            "grounding": None,
        }

    answer = generate_answer(question, hits)
    refused = is_refusal(answer)
    report = verify_citations(answer, hits)

    return {
        "answer": answer,
        # A refusal cites nothing, so don't list sources that imply support.
        "sources": [] if refused else _dedupe_sources(hits),
        "retrieved_chunks": len(hits),
        "refused": refused,
        "grounding": report.to_dict(),
    }
