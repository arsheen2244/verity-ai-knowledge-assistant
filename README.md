# Enterprise AI Knowledge Assistant

A Retrieval-Augmented Generation (RAG) system for querying organizational
documents in natural language, with traceable, cited answers.

Built to match this resume line — and to actually run:
> Developed an Enterprise AI Knowledge Assistant using a RAG pipeline to
> enable natural language querying of organizational documents. Implemented
> document processing and semantic search with vector embeddings using
> ChromaDB. Built LLM-based question answering with source citations using
> Python and FastAPI.

## Architecture

```
   .pdf/.docx/.txt
         │
         ▼
 [document_processor.py]  parse file -> raw text per page
         │
         ▼
   [chunking.py]           recursive character split (800 chars, 120 overlap)
         │
         ▼
   [embeddings.py]         local sentence-transformers model -> 384-d vector
         │                 (falls back to offline hashing vectorizer if
         │                  huggingface.co is unreachable)
         ▼
  [vector_store.py]        ChromaDB persistent collection (upsert)
                            + invalidates the BM25 keyword index


          user question
                │
                ▼
  [vector_store.py]  hybrid search:
                        1. dense search (ChromaDB, top hybrid_candidate_k)
                        2. BM25 keyword search (keyword_search.py, same k)
                        3. Reciprocal Rank Fusion merges both lists
                        4. cross-encoder reranks the merged shortlist
                           (reranker.py) down to top_k
                │
                ▼
     [llm.py]        build cited-answer prompt -> Anthropic or OpenAI
                │        (or raw-context fallback if no API key set)
                ▼
  [rag_engine.py]     assemble {answer, sources, retrieved_chunks}
                │
                ▼
     [main.py]        FastAPI: POST /api/ingest, POST /api/query


          eval/run_eval.py (separate, offline, run manually)
                │
                ▼
   runs eval/eval_questions.json through the REAL pipeline above,
   scores answers with Ragas (Faithfulness, Answer Relevancy,
   Context Precision), writes eval/eval_report.md
```

## Why it's built this way (the parts worth understanding, not memorizing)

- **Chunking is hand-rolled, not imported from LangChain.** `app/chunking.py`
  is a recursive character splitter: try paragraph breaks, then sentences,
  then words, then raw characters, whichever keeps pieces under the size
  limit while preserving as much semantic unity as possible. Overlap between
  chunks stops facts from being severed exactly at a chunk boundary.
- **Embeddings are local (sentence-transformers), not an API call.**
  Embeddings run on every chunk at ingest time and every query at search
  time — that's the highest-volume part of the system, so it's the part
  most worth keeping free and fast. It also decouples retrieval quality
  from which chat LLM you use.
- **The LLM layer is provider-agnostic.** `app/llm.py` dispatches to
  Anthropic or OpenAI based on config, and — deliberately — has a
  no-API-key fallback that returns the raw retrieved context instead of
  crashing. That's what lets `tests/test_pipeline.py` verify the entire
  retrieval pipeline (parsing, chunking, embedding, vector search) without
  needing a paid API key.
- **Citations are structural, not hoped-for.** Every chunk carries
  `{source filename, page number}` metadata from the moment it's parsed,
  all the way through embedding and storage, so the API response always
  returns a deduplicated source list alongside the answer — not just an
  LLM claiming to have cited something.

## Project layout

```
enterprise_rag/
├── app/
│   ├── config.py            # all tunable settings, env-var driven
│   ├── chunking.py          # recursive text splitter
│   ├── document_processor.py# pdf/docx/txt parsing -> Chunk objects
│   ├── embeddings.py        # local embedding model (+ offline fallback)
│   ├── vector_store.py      # ChromaDB wrapper
│   ├── llm.py                # Anthropic/OpenAI dispatch + citation prompt
│   ├── rag_engine.py         # ties retrieval + generation together
│   └── main.py                # FastAPI routes
├── static/index.html          # minimal upload + ask UI, served at /
├── tests/
│   ├── test_chunking.py
│   └── test_pipeline.py       # ingest->retrieve->answer, no API key needed
├── requirements.txt
├── .env.example
└── pytest.ini
```

## Setup

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env: set ANTHROPIC_API_KEY (or OPENAI_API_KEY + LLM_PROVIDER=openai)
# leave both blank to run in fallback mode — everything works except
# the final answer is raw context instead of an LLM-written response
```

## Run

```bash
uvicorn app.main:app --reload
```

- UI: http://127.0.0.1:8000/
- Interactive API docs: http://127.0.0.1:8000/docs

Upload a `.pdf`, `.docx`, `.txt`, or `.md` file in the UI, then ask a
question about it — you'll get an answer with `(source, page)` citations.

## Test

```bash
pytest -v
```

Six tests, all runnable offline: chunking correctness (size limits, overlap,
empty input) and a full ingest → embed → store → retrieve → answer
integration test that works even with no LLM key configured.

## Hybrid search, reranking, and eval (new)

Retrieval is now a four-stage pipeline instead of plain vector search — see
the architecture diagram above. Toggle each stage independently in `.env`:

```
hybrid_enabled=true      # dense + BM25, merged with Reciprocal Rank Fusion
rerank_enabled=true      # cross-encoder re-scores the merged shortlist
hybrid_candidate_k=20    # how many candidates each retriever contributes
rrf_k=60                 # RRF damping constant (60 is the standard default)
rerank_model_name=cross-encoder/ms-marco-MiniLM-L-6-v2
```

Why both matter: dense (vector) search is strong on paraphrase and weak on
exact terms it wasn't trained to weight (IDs, names, acronyms); BM25 keyword
search is the reverse. Reciprocal Rank Fusion combines their two ranked
lists without needing to normalize dense cosine-distance against BM25's
unbounded term-frequency scale — only rank position matters. The
cross-encoder reranker then re-scores the merged shortlist by feeding the
question and each chunk into one model *together*, which is far more
precise than comparing two independently-computed embeddings — but too slow
to run over the whole corpus, which is why it only touches the shortlist.

**To measure whether any of this actually helped:** `eval/run_eval.py` runs
your real pipeline against `eval/eval_questions.json` and scores it with
[Ragas](https://github.com/explodinggradients/ragas) on Faithfulness
(anti-hallucination), Answer Relevancy, and Context Precision (retrieval
quality specifically). Replace the example questions with real ones about
your documents, ingest those documents first, then:

```bash
python -m eval.run_eval
```

This writes `eval/eval_report.md` with per-question and averaged scores.
Turn `hybrid_enabled` / `rerank_enabled` off, rerun, and compare
Context Precision before/after — that comparison, with real numbers, is a
far stronger project artifact than "I added hybrid search."

## Running without paid API keys (Ollama or Groq) and evaluating offline

Only the answer-writing LLM was ever paid. Embeddings (`all-MiniLM-L6-v2`), the
reranker, ChromaDB and BM25 already run locally for free.

**Free hosted option (Groq):** make a free key at [console.groq.com](https://console.groq.com) and set in `.env`:

```dotenv
LLM_PROVIDER=ollama          # the name is historical: it means "any OpenAI-compatible endpoint"
OLLAMA_BASE_URL=https://api.groq.com/openai/v1
OLLAMA_API_KEY=your_groq_key
OLLAMA_MODEL=openai/gpt-oss-120b     # pick an exact name from Groq's model list
OLLAMA_MAX_TOKENS=2000               # reasoning models need more output budget
```

Free tiers are rate-limited and the model list changes. The same settings work for OpenRouter or LM Studio.


**Generate answers locally.** Install [Ollama](https://ollama.com), run
`ollama pull llama3.2:3b`, then in `.env` set `LLM_PROVIDER=ollama`. The app
talks to Ollama through its OpenAI-compatible endpoint (`OLLAMA_BASE_URL`),
so any OpenAI-compatible server works. On a CPU-only laptop expect slow answers.

**Evaluate retrieval with no LLM at all:**

```bash
python -m eval.retrieval_eval            # writes eval/retrieval_report.md
```

Label questions in `eval/eval_questions.json` with `must_contain` (a phrase
from the answering passage) or `relevant_pages`. The report compares
dense-only, BM25-only, hybrid (RRF) and hybrid + cross-encoder rerank on
Hit@k, Recall@k and MRR.

**Evaluate answer quality with Ragas, with a separate judge:**

```bash
python -m eval.run_eval                                   # generate + judge
python -m eval.run_eval --reuse-samples                   # re-judge saved answers
python -m eval.run_eval --metrics faithfulness,context_precision
```

The generator (`LLM_PROVIDER`) and the judge (`JUDGE_PROVIDER`) are separate:
generate locally, judge with a stronger model. Both are written into
`eval/eval_report.md`; always state which judge produced your scores.

## Extending it further (good next steps if you want to keep learning)

- Swap ChromaDB for pgvector/Qdrant by only touching `vector_store.py`.
- Add streaming responses (`stream=True` on the Anthropic/OpenAI call) for
  a token-by-token UI.
- Add contextual retrieval: prepend a short LLM-generated document summary
  to each chunk before embedding it, so a chunk carries context beyond its
  own boundaries — a solid next upgrade to `chunking.py`.
- Add adaptive routing: classify each question's complexity first, and only
  trigger multi-hop/agentic retrieval for the questions that actually need
  it, instead of running every question through the same fixed pipeline.
- Add auth (API key or OAuth) in front of `/api/ingest` before deploying
  anywhere real — right now anyone who can reach the server can index files.
- Track ingest provenance (who uploaded what, when) in a small SQLite table
  alongside the vector store.
