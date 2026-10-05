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

## Depends on

harness_prompts, harness_semantic, harness_schema, harness_config.

See `AGENTS.md` for the repository map and conventions.
