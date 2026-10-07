# evaluation

Scores saved runs against gold. No model calls of its own: the soft-match judge only calls a model
when the caller passes one (`make evaluate ARGS="--judge"`); the API reads the judge's cache.

The metric definitions, the reasons for each choice, how to read the numbers and their known flaws
are documented in the labelling repository: `docs/EVALUATION.md`. This README is the code map.

## Files

- `__init__.py` — `canonical_triple_key`, `score_triples` (the original exact set metric),
  `score_claims`, `evaluate_run` (every reading; adds `levels`), `evaluate_gold_row` (a gold row
  with rejected alternatives; an ambiguous question is scored AmbigQA-style, set F1 against the union of
  its accepted readings, with per-reading recall), `evaluate_attempt` (the runs sharing an attempt id
  scored as one prediction).
- `matching.py` — `score_levels`: strict / entailed / soft precision and recall, one-to-one
  (SciPy weighted assignment), locally closed DBLP scoring, vital recall, judged precision,
  and the matched / missed / extra triples for the Interpretation window.
- `relations.py` — RDFS/OWL rules for the entailed level: harness schema + DBLP seed +
  `datasets/ontology/predicate_relations.ttl`, closed once with owlrl.
- `judge.py` — `CachedJudge`: "same fact?" through the `evaluation/triple_match` prompt, cached in
  `$ARTIFACT_ROOT/judge_cache.jsonl` keyed by prompt version and pair.
- `stats.py` — percentile-bootstrap intervals over questions, paired differences.
- `benchmark.py` — `build_benchmark`: runs paired with gold by question, path and snapshot;
  per-question and complexity × path summaries (used by `/benchmark` and `scripts/evaluate.py`).
- `fewshot.py` — `fewshot_candidates`: missed vital gold triples and rejected extras, repeated
  errors first, with draft examples (`make fewshot`, `docs/PROMPTS.md`).
- `cli.py` — the older `harness-eval run` over a JSONL dataset (exact metric only).

## Settings

`HARNESS_PREDICATE_RELATIONS_FILE`, `HARNESS_JUDGE_CACHE_FILE`, `HARNESS_MODEL_JUDGE`.

## Depends on

harness_schema, harness_config, harness_prompts, harness_semantic; numpy, scipy, rdflib.

See `AGENTS.md` in this folder (entry points, imports, test, rules) and the root `AGENTS.md`.
