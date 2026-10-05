# harness_runs

Persistence of runs, claims and gold candidates for the API.

## Entry points

- `store.py` — `ArtifactStore`, `GoldCandidateStore`; `claims.py`, `relations.py`, `analysis.py`

## Imports

Layer 3 of 5 (`[tool.importlinter]` in the root `pyproject.toml`): may import the layers below (schema, config, ...), not its siblings.

## Test

```sh
uv run --python 3.12 pytest -q packages/runs
```

Repository-wide rules and the map: the root `AGENTS.md`.
