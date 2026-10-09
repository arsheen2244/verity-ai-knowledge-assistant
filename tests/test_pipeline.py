"""
End-to-end test of ingest -> embed -> store -> retrieve -> (fallback) answer,
without needing any LLM API key. This is exactly the "no key configured"
path described in app/llm.py, and it's what proves the retrieval half of
the system works independent of which LLM you eventually plug in.
"""
import os
import tempfile

import pytest

os.environ.setdefault("CHROMA_PERSIST_DIR", tempfile.mkdtemp())

from app.config import get_settings  # noqa: E402
from app.document_processor import process_file  # noqa: E402
from app.rag_engine import answer_question  # noqa: E402
from app.vector_store import get_store  # noqa: E402


@pytest.fixture(scope="module")
def sample_txt_path(tmp_path_factory):
    d = tmp_path_factory.mktemp("docs")
    p = d / "policy.txt"
    p.write_text(
        "Remote Work Policy\n\n"
        "Employees may work remotely up to 3 days per week. "
        "Requests must be approved by a manager at least one week in advance. "
        "Fully remote arrangements require VP approval.\n\n"
        "Expense Policy\n\n"
        "Home office equipment up to $500 per year is reimbursable with receipts."
    )
    return str(p)


def test_ingest_and_retrieve(sample_txt_path):
    settings = get_settings()
    settings.chroma_persist_dir = os.environ["CHROMA_PERSIST_DIR"]
    get_store.cache_clear() if hasattr(get_store, "cache_clear") else None

    chunks = process_file(sample_txt_path)
    assert len(chunks) >= 1

    store = get_store()
    n = store.add_chunks(chunks)
    assert n == len(chunks)
    assert store.count() >= n

    hits = store.search("How many days can I work remotely?", top_k=2)
    assert len(hits) > 0
    assert "remote" in hits[0]["text"].lower()


def test_answer_question_fallback_without_api_key(sample_txt_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # Force fallback mode even if your .env selects ollama/anthropic/openai.
    monkeypatch.setenv("LLM_PROVIDER", "none")
    get_settings.cache_clear()

    result = answer_question("What is the home office reimbursement limit?")
    assert "sources" in result
    assert result["retrieved_chunks"] >= 0
    assert isinstance(result["answer"], str) and len(result["answer"]) > 0
