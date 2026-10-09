"""
BM25 keyword index, the other half of hybrid search.

Why this exists: dense vector search (vector_store.py) is great at matching
*meaning* ("time off" ~ "vacation policy") but weak on exact tokens it wasn't
trained to weight highly -- product codes, IDs, acronyms, names, numbers.
BM25 is the reverse: it's a pure term-frequency algorithm, so it nails exact
keyword matches but has no notion of synonyms or paraphrase. Combining both
(see hybrid_search in vector_store.py) covers each one's blind spot.

Design choice: this index lives in memory and is rebuilt from whatever is
currently in ChromaDB, rather than being persisted separately. That keeps
ChromaDB as the single source of truth for what's indexed -- there's only
ever one place a chunk could go missing from, not two indexes that can drift
out of sync with each other.
"""
from __future__ import annotations

import re
import threading

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class KeywordIndex:
    """Thread-safe, lazily-(re)built BM25 index over a ChromaDB collection's
    current contents. Call invalidate() any time chunks are added, so the
    next search rebuilds from the latest data instead of serving stale
    results."""

    def __init__(self):
        self._lock = threading.Lock()
        self._bm25: BM25Okapi | None = None
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._metadatas: list[dict] = []
        self._dirty = True

    def invalidate(self) -> None:
        with self._lock:
            self._dirty = True

    def _rebuild(self, collection) -> None:
        data = collection.get(include=["documents", "metadatas"])
        self._ids = data["ids"]
        self._texts = data["documents"]
        self._metadatas = data["metadatas"]
        tokenized = [_tokenize(t) for t in self._texts]
        # BM25Okapi errors on an empty corpus, so guard the zero-chunk case
        # (a fresh install with nothing ingested yet) rather than crashing.
        self._bm25 = BM25Okapi(tokenized) if tokenized else None
        self._dirty = False

    def search(self, collection, query: str, top_k: int) -> list[dict]:
        with self._lock:
            if self._dirty:
                self._rebuild(collection)
            if self._bm25 is None:
                return []
            scores = self._bm25.get_scores(_tokenize(query))
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            return [
                {
                    "id": self._ids[i],
                    "text": self._texts[i],
                    "source": self._metadatas[i]["source"],
                    "page": self._metadatas[i]["page"],
                    "bm25_score": float(scores[i]),
                }
                for i in ranked
                if scores[i] > 0  # a zero score means no query term appeared at all
            ]


_index: KeywordIndex | None = None


def get_keyword_index() -> KeywordIndex:
    global _index
    if _index is None:
        _index = KeywordIndex()
    return _index
