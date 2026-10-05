# harness_ledger

The coverage ledger: obligations a run must settle before its graph is queried.

## Entry points

- `__init__.py` — `Ledger`, `Obligation`

## Imports

Layer 4 of 5 (`[tool.importlinter]` in the root `pyproject.toml`): may import harness_schema and harness_config only.

## Test

```sh
uv run --python 3.12 pytest -q packages/ledger
```

Repository-wide rules and the map: the root `AGENTS.md`.
