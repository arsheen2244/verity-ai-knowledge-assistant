"""
FastAPI backend. Three endpoints:
  POST /api/ingest  -- upload a .pdf/.docx/.txt/.md file, it gets chunked+embedded+stored
  POST /api/query   -- ask a question, get a cited answer
  GET  /api/health  -- sanity check + how many chunks are indexed

Run with:  uvicorn app.main:app --reload
Docs at:   http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import os
import shutil
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .document_processor import process_file
from .rag_engine import answer_question
from .vector_store import get_store

app = FastAPI(title="Enterprise AI Knowledge Assistant", version="1.0.0")

ALLOWED_EXT = {".pdf", ".docx", ".txt", ".md"}


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None


class Source(BaseModel):
    source: str
    page: int


class Grounding(BaseModel):
    verified: bool
    citations_found: int
    unverified_citations: list[str]
    quotes_found: int
    unverified_quotes: list[str]


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    retrieved_chunks: int
    refused: bool = False
    grounding: Grounding | None = None


class IngestResponse(BaseModel):
    filename: str
    chunks_indexed: int
    total_chunks_in_store: int


@app.get("/api/health")
def health():
    return {"status": "ok", "indexed_chunks": get_store().count()}


@app.post("/api/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXT)}")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        chunks = process_file(tmp_path)
        # Re-tag with the real uploaded filename (tempfile has a random name)
        for c in chunks:
            c.source = file.filename
            c.chunk_id = c.chunk_id.replace(os.path.basename(tmp_path), file.filename)
        store = get_store()
        n = store.add_chunks(chunks)
        return IngestResponse(filename=file.filename, chunks_indexed=n, total_chunks_in_store=store.count())
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        os.unlink(tmp_path)


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(400, "question must not be empty")
    try:
        result = answer_question(request.question, top_k=request.top_k)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))


# Serve the tiny demo UI at /
if os.path.isdir(os.path.join(os.path.dirname(__file__), "..", "static")):
    app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "..", "static"), html=True), name="static")
