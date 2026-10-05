# 0005 Big functions are split behind their current names

**Context.** A few functions carried most of the complexity (radon, 2026-10-05): `_fact_query` 183,
`_retrieve` 109, `_success` 102, `_run_agent_async` 88, `score_levels` 71, `CorpusStore.aggregate` 65,
`extract_edc` 56. A change anywhere in them meant reading all of them.

**Decision.** Split by responsibility (one function per fact operation, one module per extraction stage, steps
for retrieval and the run result, policies for the agent loop). Public names and imports stay as they are.

**Consequences.** No caller changes. Each split is its own commit, with the complexity before and after in the
message. Scoring is split only when `make evaluate` gives the same numbers on the same runs.
