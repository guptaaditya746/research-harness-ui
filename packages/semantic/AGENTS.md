# harness_semantic

The ontology: the DBLP seed, the CS-KG content core, run-local terms, SHACL validation and OWL closure.

## Entry points

- `__init__.py` — `hand_written_subschema`, `term_origin`, `validate_subschema`, `owlrl_closure`

## Imports

Layer 4 of 5 (`[tool.importlinter]` in the root `pyproject.toml`): may import harness_schema and harness_config only.

## Test

```sh
uv run --python 3.12 pytest -q packages/semantic
```

## Rules

- Change `ONTOLOGY_VERSION` with any change to the seed or the CS-KG subset.

Repository-wide rules and the map: the root `AGENTS.md`.
