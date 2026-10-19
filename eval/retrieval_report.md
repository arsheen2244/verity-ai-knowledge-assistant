# Retrieval evaluation report (no LLM used)

Generated: 2026-10-09T10:17:37+00:00Z
Labeled questions: 50 (skipped 5 without labels)
Chunks indexed: 82
Candidates per retriever: 20, RRF k: 60
Cross-encoder loaded: True

| Config | hit@1 | hit@3 | hit@5 | recall@1 | recall@3 | recall@5 | mrr |
|---|---|---|---|---|---|---|---|
| dense | 0.60 | 0.88 | 0.96 | 0.60 | 0.88 | 0.96 | 0.74 |
| bm25 | 0.50 | 0.86 | 0.86 | 0.50 | 0.86 | 0.86 | 0.66 |
| hybrid | 0.66 | 0.86 | 0.94 | 0.66 | 0.86 | 0.94 | 0.77 |
| hybrid+rerank | 0.74 | 0.94 | 0.96 | 0.74 | 0.94 | 0.96 | 0.83 |

## Questions where hybrid+rerank found nothing relevant in the top 5

- What SQL command deletes an entire database and all its tables?
- How do I find rows where a column has no value?

Hit@k: a relevant chunk appears in the top k. Recall@k: share of labeled targets covered. MRR: mean of 1/rank of the first relevant chunk. With a small question set these numbers are indicative, not statistically strong -- report the question count alongside them.