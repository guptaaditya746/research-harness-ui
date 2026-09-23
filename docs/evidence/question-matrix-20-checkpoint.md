# 20-type question matrix: first pass

The [manifest](../../datasets/question_matrix_20.json) covers 20 distinct question types. The planned matrix has three paths and three repetitions per type (180 runs). The [first-pass run file](question-matrix-20-pre-retrieval-fix-2026-09-23.json) contains 132 completed runs through the first reported-metric type.

- Metadata, positive citation, and aggregation questions completed consistently after routing verified corpus facts directly into atomic triples.
- The absent incoming-citation question returned `no_results` in all nine runs, as expected from the resolved corpus references.
- The first open-ended metric question retrieved 78 passages in adaptive mode, requiring 10 extraction batches. One fast run returned the supported 41.5% CALVIN result; five other fast/adaptive runs ended with provider errors while the small-model queue was busy.
- Agentic runs used the explicitly marked degraded fallback because the heavy model remained queued. They must not be counted as completed agentic evaluations.

The pass was stopped before the remaining synthesis and ambiguity questions. The next run will use evidence ranking and staged extraction to reduce irrelevant model input without imposing a fixed triple-count cap. Completion status here is not a precision or recall score; the 20 questions do not yet have approved gold triples.
