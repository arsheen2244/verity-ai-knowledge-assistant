"""
Thin wrapper around ChromaDB so the rest of the app never talks to Chroma
directly. This isolation matters: if you ever swap ChromaDB for Pinecone,
Qdrant, or pgvector, only this file changes.

We pass in our own embeddings (from embeddings.py) rather than letting
Chroma call an embedding function itself -- that keeps embedding logic in
one obvious place instead of split across two files.

Retrieval is now a three-stage pipeline instead of plain vector search:
  1. Dense search (ChromaDB)   -- catches semantic/paraphrase matches
  2. BM25 keyword search       -- catches exact terms dense search misses
  3. Reciprocal Rank Fusion    -- merges both ranked lists into one
  4. Cross-encoder reranking   -- re-scores the merged shortlist precisely
Each stage is optional and controlled by settings.hybrid_enabled /
settings.rerank_enabled, so you can turn either off and fall back toward
the original pure-dense-search behavior for comparison or debugging.
"""
from __future__ import annotations

import chromadb

from .config import get_settings
from .document_processor import Chunk
from .embeddings import embed_texts
from .keyword_search import get_keyword_index
from .reranker import rerank


class VectorStore:
    def __init__(self):
        settings = get_settings()
        self._client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self._collection = self._client.get_or_create_collection(name=settings.collection_name)

    def add_chunks(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        vectors = embed_texts([c.text for c in chunks])
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "page": c.page} for c in chunks],
        )
        # The BM25 index snapshotted the collection's old contents; tell it
        # to rebuild on next search now that new chunks have landed.
        get_keyword_index().invalidate()
        return len(chunks)

    def _dense_search(self, query: str, top_k: int) -> list[dict]:
        query_vector = embed_texts([query])[0]
        results = self._collection.query(query_embeddings=[query_vector], n_results=top_k)
        hits = []
        if results["documents"] and results["documents"][0]:
            for id_, text, meta, dist in zip(
                results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
            ):
                hits.append({"id": id_, "text": text, "source": meta["source"], "page": meta["page"], "distance": dist})
        return hits

    def _reciprocal_rank_fusion(self, ranked_lists: list[list[dict]], rrf_k: int) -> list[dict]:
        """Merges several ranked hit lists into one by Reciprocal Rank
        Fusion: RRF_score(doc) = sum(1 / (rrf_k + rank)) across every list it
        appears in. Unlike averaging raw scores, RRF needs no score
        normalization between dense cosine-distance and BM25's unbounded
        term-frequency scale -- only rank position matters, which is what
        makes it the standard way to combine two differently-scaled
        retrievers."""
        fused: dict[str, float] = {}
        by_id: dict[str, dict] = {}
        for ranked_list in ranked_lists:
            for rank, hit in enumerate(ranked_list):
                fused[hit["id"]] = fused.get(hit["id"], 0.0) + 1.0 / (rrf_k + rank)
                by_id.setdefault(hit["id"], hit)  # keep the first-seen copy for its text/metadata
        ordered_ids = sorted(fused, key=lambda id_: fused[id_], reverse=True)
        return [{**by_id[id_], "rrf_score": fused[id_]} for id_ in ordered_ids]

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        settings = get_settings()
        k = top_k or settings.top_k

        if not settings.hybrid_enabled:
            hits = self._dense_search(query, k)
        else:
            candidate_k = settings.hybrid_candidate_k
            dense_hits = self._dense_search(query, candidate_k)
            keyword_hits = get_keyword_index().search(self._collection, query, candidate_k)
            hits = self._reciprocal_rank_fusion([dense_hits, keyword_hits], settings.rrf_k)

        if settings.rerank_enabled:
            hits = rerank(query, hits, k)
        else:
            hits = hits[:k]

        return hits

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        settings = get_settings()
        self._client.delete_collection(settings.collection_name)
        self._collection = self._client.get_or_create_collection(name=settings.collection_name)


_store: VectorStore | None = None


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
