# extraction

Structured model calls and the extraction engine. The JSON model client (retries, cancellation, progress events, prompt versions) and the steps shared by all paths: schema inference, extraction, review, answer, claims.

## Files

- `model_client.py` — `OpenAICompatibleModel`, `complete(model, prompt)`
- `call_control.py` — run deadline, stop, interruptible calls
- `events.py` — `model_purpose`, `model_events`, `using_prompt`
- `engine.py` — `ExtractionEngine`: the run's schema (run-local classes and relations), signatures, axioms
- `stage_schema.py`, `stage_extract.py`, `stage_review.py`, `stage_answer.py` — the stages, mixins of the engine (schema inference; direct and EDC extraction; evidence review; answer composition and verification)
- `naming.py`, `citations.py` — relation names and prompt shape; answer citations
- `schemas.py` — reply models
- `claims.py` — claim subgraphs (imported by labelling; keep the names)

## EDC extraction (`stage_extract.extract_edc`)

Extract (open propositions) → Define (one definition per relation phrase) → Canonicalize: a lexical
score proposes the top 3 properties per proposition, and one `edc_map` call per batch lets the model
choose a property or none for every proposition (confidence ≥ `HARNESS_MAP_MODEL_MIN`). A Paper-domain
property keeps a proposition only when its subject is the paper (its id, its exact title, or "we",
"this paper", ...), written as the paper id; otherwise `domain_mismatch`. A paper as object under a range
that cannot hold a paper is `inverted_direction`. Relations kept with `HARNESS_UNMAPPED_RELATIONS=keep`
get the generic ResearchEntity domain and keep their subjects. The `canonicalize` report gives each
proposition's method.

Deviations from EDC (Zhang & Soh, EMNLP 2024): lexical candidates plus an LLM choice over the full
catalog instead of definition embeddings or a trained retriever; Define without passage context; no EDC+R
refinement; self-canonicalization only by exact name (not merged); batches of 1–2 passages; zero-shot
Extract by default; the best lexical candidate alone only when the mapping call fails (`lexical_fallback`).

## Depends on

harness_prompts, harness_semantic, harness_schema, harness_config.

See `AGENTS.md` in this folder (entry points, imports, test, rules) and the root `AGENTS.md`.
