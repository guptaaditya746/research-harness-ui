# harness_corpus

The corpus store and one-time ingest: papers, passages, DBLP facts, references; exact search, BM25 and dense ranking.

## Entry points

- `store.py` — `CorpusStore` (identifiers, ranked retrieval); `store_tools.py` — the direct-read tools; `aggregates.py`
- `dsl.py`, `rows.py`, `dense.py`; `ingest.py`, `tei.py`, `dblp.py`, `__main__.py`

## Imports

Layer 4 of 5 (`[tool.importlinter]` in the root `pyproject.toml`): may import harness_schema and harness_config only.

## Test

```sh
uv run --python 3.12 pytest -q packages/corpus
```

## Rules

- Tool results keep their ids (passage, fact, reference): agents may only cite ids they were shown.

Repository-wide rules and the map: the root `AGENTS.md`.
