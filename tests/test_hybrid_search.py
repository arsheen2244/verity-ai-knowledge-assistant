"""
Tests the Reciprocal Rank Fusion merge in isolation from embeddings/ChromaDB,
since RRF is pure ranking arithmetic and doesn't need a real vector store or
network access to test correctly.
"""
from app.vector_store import VectorStore


def _store() -> VectorStore:
    # __new__ skips __init__, so no ChromaDB client/persist dir is touched --
    # we only want the _reciprocal_rank_fusion method, which has no
    # dependency on self.
    return VectorStore.__new__(VectorStore)


def test_rrf_favors_docs_ranked_highly_in_both_lists():
    store = _store()
    dense = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    keyword = [{"id": "b"}, {"id": "a"}, {"id": "d"}]

    fused = store._reciprocal_rank_fusion([dense, keyword], rrf_k=60)
    fused_ids = [h["id"] for h in fused]

    # "a" and "b" each appear near the top of BOTH lists, so RRF should rank
    # them above "c" and "d", which each only appear in one list.
    assert set(fused_ids[:2]) == {"a", "b"}


def test_rrf_includes_docs_found_by_only_one_retriever():
    store = _store()
    dense = [{"id": "x"}]
    keyword = [{"id": "y"}]

    fused = store._reciprocal_rank_fusion([dense, keyword], rrf_k=60)

    # A document doesn't need to appear in both lists to survive the merge --
    # this is what lets hybrid search catch a keyword-only match dense search
    # missed entirely, or vice versa.
    assert {h["id"] for h in fused} == {"x", "y"}


def test_rrf_handles_empty_lists():
    store = _store()
    assert store._reciprocal_rank_fusion([[], []], rrf_k=60) == []
