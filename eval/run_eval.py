"""
Ragas evaluation harness (answer quality, judged by an LLM).

What this measures, and why it beats "it works when I tried it": it runs a
fixed set of golden questions through the REAL pipeline and scores the
answers on three axes:

  Faithfulness      -- does the ANSWER only claim things the retrieved
                        chunks actually say? (the anti-hallucination score)
  Answer Relevancy   -- does the answer actually address the question asked?
  Context Precision  -- of the chunks retrieval fetched, how many were
                        relevant? (the one your hybrid search / reranking
                        should move)

Two separate models are involved, and they do NOT have to be the same:
  * the GENERATOR writes the answers      (LLM_PROVIDER in .env)
  * the JUDGE grades them                 (JUDGE_PROVIDER in .env)
Small local models (Ollama on a laptop CPU) are fine generators but noisy
judges, so a good setup is: generate locally, judge with something stronger.
The README line should always say which was which.

No API key at all? Use eval/retrieval_eval.py instead -- it measures the
retrieval stages with plain ranking metrics and needs no LLM.

Usage:
    1. Start nothing -- but ingest your real documents first (UI or
       POST /api/ingest). This script evaluates whatever is in the store.
    2. Put real questions in eval/eval_questions.json.
    3. python -m eval.run_eval                       # generate + judge
       python -m eval.run_eval --reuse-samples       # re-judge saved answers
       python -m eval.run_eval --metrics faithfulness,context_precision
    4. Read eval/eval_report.md.

Generating answers with a slow local model is the expensive part, so answers
are saved to eval/samples.json. Re-run with --reuse-samples to change the
judge without regenerating anything.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.llm import generate_answer  # noqa: E402
from app.vector_store import get_store  # noqa: E402

QUESTIONS_PATH = Path(__file__).parent / "eval_questions.json"
SAMPLES_PATH = Path(__file__).parent / "samples.json"
REPORT_PATH = Path(__file__).parent / "eval_report.md"

METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision")


def _load_questions() -> list[dict]:
    data = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    return data["questions"]


def _run_pipeline(questions: list[dict]) -> list[dict]:
    """Runs each question through the real retrieval + generation path ONCE
    and keeps exactly the chunks the answer was generated from -- so the
    contexts Ragas grades are the contexts the model actually saw."""
    store = get_store()
    samples = []
    for i, item in enumerate(questions, 1):
        question = item["question"]
        print(f"  [{i}/{len(questions)}] {question[:70]}")
        hits = store.search(question)
        answer = generate_answer(question, hits)
        samples.append(
            {
                "user_input": question,
                "response": answer,
                "retrieved_contexts": [h["text"] for h in hits] or [""],
                "reference": item.get("ground_truth", ""),
            }
        )
    return samples


def _resolved_judge_provider() -> str:
    s = get_settings()
    return s.llm_provider if s.judge_provider == "same" else s.judge_provider


def _judge_description() -> str:
    s = get_settings()
    p = _resolved_judge_provider()
    model = {
        "anthropic": s.judge_model or s.anthropic_model,
        "openai": s.judge_model or s.openai_model,
        "ollama": s.judge_model or s.ollama_model,
        "custom": s.judge_model,
    }.get(p, "unknown")
    return f"{p} / {model}"


def _generator_description() -> str:
    s = get_settings()
    model = {
        "anthropic": s.anthropic_model,
        "openai": s.openai_model,
        "ollama": s.ollama_model,
    }.get(s.llm_provider, "none (raw-context fallback)")
    return f"{s.llm_provider} / {model}"


def _build_judge_chat_model():
    """Builds the LangChain chat model Ragas uses to grade answers."""
    s = get_settings()
    provider = _resolved_judge_provider()

    if provider == "anthropic":
        if not s.anthropic_api_key:
            raise RuntimeError("JUDGE_PROVIDER resolves to 'anthropic' but ANTHROPIC_API_KEY is not set.")
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=s.judge_model or s.anthropic_model, api_key=s.anthropic_api_key, temperature=0)

    if provider == "openai":
        if not s.openai_api_key:
            raise RuntimeError("JUDGE_PROVIDER resolves to 'openai' but OPENAI_API_KEY is not set.")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=s.judge_model or s.openai_model, api_key=s.openai_api_key, temperature=0)

    if provider == "ollama":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=s.judge_model or s.ollama_model,
            base_url=s.ollama_base_url,
            api_key=s.ollama_api_key,
            temperature=0,
            timeout=300,
        )

    if provider == "custom":
        if not (s.judge_base_url and s.judge_model):
            raise RuntimeError("JUDGE_PROVIDER=custom needs JUDGE_BASE_URL and JUDGE_MODEL (and usually JUDGE_API_KEY).")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=s.judge_model,
            base_url=s.judge_base_url,
            api_key=s.judge_api_key or "none",
            temperature=0,
            timeout=300,
        )

    raise RuntimeError(
        f"Ragas needs a real judge LLM (got provider '{provider}'). Set JUDGE_PROVIDER to "
        "anthropic / openai / ollama / custom in .env -- or use `python -m eval.retrieval_eval`, "
        "which needs no LLM at all."
    )


def _build_ragas_llm_and_embeddings():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    settings = get_settings()
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model_name)
    return LangchainLLMWrapper(_build_judge_chat_model()), LangchainEmbeddingsWrapper(embeddings)


def _select_metrics(names: list[str]):
    from ragas.metrics import AnswerRelevancy, ContextPrecision, Faithfulness

    available = {
        "faithfulness": Faithfulness,
        "answer_relevancy": AnswerRelevancy,
        "context_precision": ContextPrecision,
    }
    unknown = [n for n in names if n not in available]
    if unknown:
        raise SystemExit(f"Unknown metric(s) {unknown}. Choose from: {', '.join(METRIC_NAMES)}")
    return [available[n]() for n in names]


def _write_report(scores_df, n_questions: int, metric_names: list[str]) -> None:
    mean_scores = scores_df.mean(numeric_only=True)
    scored_counts = scores_df[list(mean_scores.index)].notna().sum()
    lines = [
        "# Ragas evaluation report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}Z",
        f"Questions evaluated: {n_questions}",
        f"Answer generator: {_generator_description()}",
        f"Judge model: {_judge_description()}",
        f"Metrics: {', '.join(metric_names)}",
        f"Retrieval: hybrid={get_settings().hybrid_enabled}, rerank={get_settings().rerank_enabled}, top_k={get_settings().top_k}",
        "",
        "| Metric | Score (0.0-1.0) | Questions scored |",
        "|---|---|---|",
    ]
    for metric, score in mean_scores.items():
        lines.append(f"| {metric} | {score:.2f} | {int(scored_counts[metric])} of {n_questions} |")
    if (scored_counts < n_questions).any():
        lines += [
            "",
            "WARNING: some judge calls failed (rate limits / timeouts), so the scores above "
            "cover only the questions that succeeded. Re-run with --reuse-samples to fill the "
            "gaps before quoting these numbers.",
        ]
    lines += [
        "",
        "Faithfulness is the anti-hallucination check. Context Precision reflects "
        "retrieval quality (hybrid search, reranking) -- if it is low, the fix is in "
        "vector_store.py, not llm.py. Scores from a small local judge are noisy; "
        "treat them as indicative and say which judge produced them.",
        "",
        "## Per-question scores",
        "",
        scores_df.to_markdown(index=False),
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ragas evaluation for the RAG pipeline")
    parser.add_argument("--reuse-samples", action="store_true", help="skip generation; judge the answers saved in eval/samples.json")
    parser.add_argument("--metrics", default=",".join(METRIC_NAMES), help=f"comma-separated subset of: {', '.join(METRIC_NAMES)}")
    parser.add_argument("--limit", type=int, default=0, help="judge only N evenly-spaced samples (saves tokens on rate-limited judges)")
    args = parser.parse_args()
    metric_names = [m.strip() for m in args.metrics.split(",") if m.strip()]

    if args.reuse_samples:
        if not SAMPLES_PATH.exists():
            raise SystemExit("eval/samples.json not found -- run once without --reuse-samples first.")
        samples = json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))
        print(f"Loaded {len(samples)} saved answers from {SAMPLES_PATH.name}")
    else:
        questions = _load_questions()
        if not questions or questions[0]["question"].startswith("Example:"):
            print(
                "eval_questions.json still has the example/placeholder questions. Replace "
                "them with real questions about your ingested documents first.",
                file=sys.stderr,
            )
        store_count = get_store().count()
        if store_count == 0:
            raise RuntimeError("The vector store is empty -- ingest documents first (UI or POST /api/ingest).")
        print(f"Generating answers for {len(questions)} questions ({store_count} chunks indexed)...")
        samples = _run_pipeline(questions)
        SAMPLES_PATH.write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved answers to {SAMPLES_PATH.name} (re-judge later with --reuse-samples)")

    if args.limit and args.limit < len(samples):
        step = max(1, len(samples) // args.limit)
        samples = samples[::step][: args.limit]
        print(f"--limit: judging {len(samples)} evenly-spaced samples")

    from ragas import EvaluationDataset, evaluate
    from ragas.run_config import RunConfig

    llm, embeddings = _build_ragas_llm_and_embeddings()
    dataset = EvaluationDataset.from_list(samples)

    print(f"Scoring with Ragas, judge = {_judge_description()} ...")
    result = evaluate(
        dataset=dataset,
        metrics=_select_metrics(metric_names),
        llm=llm,
        embeddings=embeddings,
        # One worker + long timeout keeps slow local / rate-limited judges from failing mid-run.
        run_config=RunConfig(timeout=300, max_workers=1 if _resolved_judge_provider() in ("ollama", "custom") else 4),
    )

    scores_df = result.to_pandas()
    failed = [m for m in scores_df.select_dtypes("number").columns if scores_df[m].isna().all()]
    if failed:
        raise SystemExit(
            f"Every judge call failed for: {', '.join(failed)}. Usually a rate limit (check the "
            "provider's daily token limit) -- wait, or use --limit / a smaller question set, then "
            "re-run with --reuse-samples. No report was written."
        )
    _write_report(scores_df, len(samples), metric_names)
    print(f"\nDone. Report written to {REPORT_PATH}")
    print(scores_df.mean(numeric_only=True).to_string())


if __name__ == "__main__":
    main()