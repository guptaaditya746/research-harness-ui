# harness_extraction

Structured model calls and the extraction engine shared by every path: schema inference, extraction (direct and EDC), evidence review, answer composition and verification, claims.

## Entry points

- `engine.py` — `ExtractionEngine` (the run's schema); stages in `stage_schema.py`, `stage_extract.py`, `stage_review.py`, `stage_answer.py`
- `model_client.py` — `complete(model, prompt)`; `call_control.py`, `events.py`
- `claims.py` — claim subgraphs (labelling imports `build_claims`: keep the name)

## Imports

Layer 3 of 5 (`[tool.importlinter]` in the root `pyproject.toml`): may import harness_corpus, harness_semantic, harness_observability, harness_ledger, harness_prompts, harness_schema, harness_config (not its siblings graphstore, mcp_corpus, evaluation, runs).

## Test

```sh
uv run --python 3.12 pytest -q packages/extraction
```

## Rules

- Every model call renders a named prompt (`harness_prompts.render`); no prompt text in Python.
- Reply models in `schemas.py` with `extra="forbid"` reject unknown keys: change the prompt and the model together.

Repository-wide rules and the map: the root `AGENTS.md`.
