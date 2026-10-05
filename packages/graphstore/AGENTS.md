# harness_graphstore

The run's provenance dataset (named graphs per event) and its SPARQL checks.

## Entry points

- `__init__.py` — `ProvenanceDataset`, `sparql_check`

## Imports

Layer 3 of 5 (`[tool.importlinter]` in the root `pyproject.toml`): may import the layers below (ledger, semantic, ...), not its siblings.

## Test

```sh
uv run --python 3.12 pytest -q packages/graphstore
```

Repository-wide rules and the map: the root `AGENTS.md`.
