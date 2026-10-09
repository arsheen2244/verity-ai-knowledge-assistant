"""
LLM generation layer.

Design choice: an abstract `generate_answer(question, context_blocks)`
function that dispatches to Anthropic, OpenAI, or a local Ollama server
based on config, plus a
"none" fallback that returns the raw retrieved context. That fallback
matters more than it looks: it means you can demo and test the entire
retrieval half of this system (chunking, embedding, vector search) with
zero API keys and zero cost, which is exactly what we do in tests/.
"""
from __future__ import annotations

from tenacity import retry, stop_after_attempt, wait_exponential

from .config import get_settings

SYSTEM_PROMPT = (
    "You are an Enterprise AI Knowledge Assistant. Answer the user's question "
    "using ONLY the provided context below. If the answer is not contained in "
    "the context, say \"I don't know based on the available documents.\" "
    "For every factual claim, cite the source in the form (source, page X)."
)


def _build_context_block(hits: list[dict]) -> str:
    blocks = []
    for h in hits:
        blocks.append(f"[Source: {h['source']}, Page {h['page']}]\n{h['text']}")
    return "\n\n---\n\n".join(blocks)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def _call_anthropic(question: str, context: str) -> str:
    import anthropic

    settings = get_settings()
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    resp = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=800,
        system=SYSTEM_PROMPT + f"\n\n--- CONTEXT ---\n{context}\n--- END CONTEXT ---",
        messages=[{"role": "user", "content": question}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def _call_openai(question: str, context: str) -> str:
    from openai import OpenAI

    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    resp = client.chat.completions.create(
        model=settings.openai_model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT + f"\n\n--- CONTEXT ---\n{context}\n--- END CONTEXT ---"},
            {"role": "user", "content": question},
        ],
    )
    return resp.choices[0].message.content


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=4), reraise=True)
def _call_ollama(question: str, context: str) -> str:
    """Local / OpenAI-compatible generation. No paid key needed. On a CPU-only
    laptop this is slow (seconds to a minute per answer) but free."""
    from openai import OpenAI

    settings = get_settings()
    client = OpenAI(base_url=settings.ollama_base_url, api_key=settings.ollama_api_key, timeout=300)
    resp = client.chat.completions.create(
        model=settings.ollama_model,
        temperature=0.0,
        max_tokens=settings.ollama_max_tokens,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT + f"\n\n--- CONTEXT ---\n{context}\n--- END CONTEXT ---"},
            {"role": "user", "content": question},
        ],
    )
    return resp.choices[0].message.content or ""


def generate_answer(question: str, hits: list[dict]) -> str:
    settings = get_settings()
    context = _build_context_block(hits)

    if not hits:
        return "I don't know based on the available documents. (No relevant chunks were retrieved -- have you ingested any documents yet?)"

    if settings.llm_provider == "anthropic" and settings.anthropic_api_key:
        return _call_anthropic(question, context)
    if settings.llm_provider == "openai" and settings.openai_api_key:
        return _call_openai(question, context)

    if settings.llm_provider == "ollama":
        try:
            return _call_ollama(question, context)
        except Exception as e:  # noqa: BLE001 -- turn connection/model errors into one clear message
            raise RuntimeError(
                f"Could not get an answer from {settings.ollama_base_url} "
                f"using model '{settings.ollama_model}': {e}. For local Ollama: is it running, and did you "
                f"run `ollama pull {settings.ollama_model}`? For Groq or another hosted API: check "
                f"OLLAMA_BASE_URL, OLLAMA_API_KEY and the exact model name."
            ) from e

    # No API key configured: fall back to a transparent "raw retrieval" mode
    # so the rest of the pipeline stays testable and demo-able.
    return (
        "[No LLM API key configured -- showing raw retrieved context instead of a generated answer]\n\n"
        + context
    )
