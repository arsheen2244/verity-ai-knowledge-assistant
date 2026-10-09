# Retrieval evaluation report (no LLM used)

Generated: 2026-10-09T06:50:03+00:00Z
Labeled questions: 50 (skipped 5 without labels)
Chunks indexed: 82
Candidates per retriever: 20, RRF k: 60
Cross-encoder loaded: True

| Config | hit@1 | hit@3 | hit@5 | recall@1 | recall@3 | recall@5 | mrr |
|---|---|---|---|---|---|---|---|
| dense | 0.44 | 0.66 | 0.74 | 0.44 | 0.66 | 0.74 | 0.56 |
| bm25 | 0.36 | 0.68 | 0.68 | 0.36 | 0.68 | 0.68 | 0.50 |
| hybrid | 0.52 | 0.68 | 0.72 | 0.52 | 0.68 | 0.72 | 0.60 |
| hybrid+rerank | 0.60 | 0.74 | 0.74 | 0.60 | 0.74 | 0.74 | 0.66 |

## Questions where hybrid+rerank found nothing relevant in the top 5

- What SQL command deletes an entire database and all its tables?
- How can I move a column to the first position of a table?
- How do I find rows where a column has no value?
- What happens if I run UPDATE without a WHERE clause?
- What is the difference between DELETE FROM users and DROP TABLE users?
- How can I calculate a user's age in years?
- What is the default mode for transactions in MySQL and how do I turn it off?
- How is a PRIMARY KEY different from a UNIQUE constraint?
- How do I create a foreign key that links addresses to users?
- What does an INNER JOIN return?
- What is the difference between UNION and UNION ALL?
- Does a MySQL view store its own data?
- What is the difference between WHERE and HAVING?

Hit@k: a relevant chunk appears in the top k. Recall@k: share of labeled targets covered. MRR: mean of 1/rank of the first relevant chunk. With a small question set these numbers are indicative, not statistically strong -- report the question count alongside them.