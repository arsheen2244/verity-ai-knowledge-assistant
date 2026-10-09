"""
Central configuration for the Enterprise AI Knowledge Assistant.

Why this file exists (learning note):
Hard-coding model names, chunk sizes, and API keys throughout the codebase
makes a project brittle. Every production RAG system centralizes its knobs
in one place so you can tune retrieval quality without hunting through files.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM provider ---
    # "anthropic", "openai", or "ollama" (local, free, no API key). You only need
    # ONE provider configured. Any other value (e.g. "none") runs the app in
    # fallback mode and returns the raw retrieved context.
    llm_provider: str = "anthropic"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-4-6"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"

    # Ollama exposes an OpenAI-compatible API, so the same client works for
    # it. The base URL is configurable, so this can also point at any other
    # OpenAI-compatible server (a free hosted tier, LM Studio, vLLM, ...).
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "llama3.2:3b"
    ollama_api_key: str = "ollama"  # required by the client, ignored by Ollama
    # Output-token cap for this OpenAI-compatible path. Reasoning models (e.g.
    # openai/gpt-oss-120b on Groq) spend part of it on hidden reasoning, so
    # raise this (e.g. 2000) if answers come back empty or cut off.
    ollama_max_tokens: int = 600

    # --- Ragas "judge" model (eval/run_eval.py only) ---
    # The model that GRADES answers is a separate concern from the model that
    # WRITES them. Small local models are unreliable graders, so you can keep
    # generating with Ollama and judge with something stronger.
    #   "same"       -> reuse llm_provider above
    #   "anthropic" / "openai" / "ollama" -> use that provider's settings above
    #   "custom"     -> any OpenAI-compatible endpoint (judge_base_url/api_key/model)
    judge_provider: str = "same"
    judge_base_url: str | None = None
    judge_api_key: str | None = None
    judge_model: str | None = None

    # --- Embeddings ---
    # Local, free, no API key needed. Runs on CPU. 384-dim vectors.
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Chunking ---
    chunk_size: int = 800        # characters, not tokens (simple + good enough to learn on)
    chunk_overlap: int = 120     # ~15% overlap so context isn't severed at chunk edges

    # --- Retrieval ---
    top_k: int = 4

    # --- Hybrid search (dense + keyword) ---
    # Dense (ChromaDB) and BM25 (keyword) each retrieve `hybrid_candidate_k`
    # candidates, merged by Reciprocal Rank Fusion, before the reranker cuts
    # the merged list down to `top_k`. Set hybrid_enabled=False to fall back
    # to pure dense search (the old behavior).
    hybrid_enabled: bool = True
    hybrid_candidate_k: int = 20
    rrf_k: int = 60  # standard RRF damping constant; higher = flatter weighting of rank

    # --- Reranking ---
    # Cross-encoders score a (question, chunk) pair jointly, which is far
    # more precise than comparing two independently-computed embeddings --
    # but too slow to run over a whole corpus, which is why it only re-scores
    # the small hybrid_candidate_k shortlist, not every chunk in the store.
    rerank_enabled: bool = True
    rerank_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Storage ---
    chroma_persist_dir: str = "./chroma_db"
    collection_name: str = "enterprise_docs"


@lru_cache
def get_settings() -> Settings:
    """Cached so we parse env vars once per process, not on every request."""
    return Settings()
