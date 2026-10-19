"""
Wipes the vector store and re-ingests one document with the CURRENT settings
(.env), so you can change CHUNK_SIZE / EMBEDDING_MODEL_NAME and re-measure
without clicking through the UI.

Usage:  python -m eval.reingest "path\\to\\MySQL Handbook.pdf"

Keep the file name identical to the one you originally uploaded, so page
labels in eval_questions.json (if you use relevant_pages) still match.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.document_processor import process_file  # noqa: E402
from app.vector_store import get_store  # noqa: E402


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python -m eval.reingest "path/to/document.pdf"')
    path = sys.argv[1]
    name = os.path.basename(path)

    s = get_settings()
    print(f"chunk_size={s.chunk_size} overlap={s.chunk_overlap} embeddings={s.embedding_model_name}")

    chunks = process_file(path)
    for c in chunks:
        c.source = name
        c.chunk_id = c.chunk_id.replace(os.path.basename(path), name)

    store = get_store()
    store.reset()
    n = store.add_chunks(chunks)
    print(f"Indexed {n} chunks from {name}. Store now holds {store.count()}.")


if __name__ == "__main__":
    main()