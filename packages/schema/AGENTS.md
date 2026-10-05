# harness_schema

The run model shared by the API, the UI (generated TypeScript) and labelling.

## Entry points

- `models.py` — `Run`, `Reading`, `Triple`, `required_artifacts`; `generate.py` — the TypeScript types

## Imports

Layer 5 of 5 (`[tool.importlinter]` in the root `pyproject.toml`): may import nothing from the harness.

## Test

```sh
uv run --python 3.12 pytest -q packages/schema
```

## Rules

- After a model change run `make fixtures` (regenerates `web/src/generated/`); labelling imports these models.

Repository-wide rules and the map: the root `AGENTS.md`.
