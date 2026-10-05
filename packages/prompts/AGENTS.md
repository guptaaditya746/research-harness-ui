# harness_prompts

Every prompt as a versioned Jinja template (one set, v2).

## Entry points

- `__init__.py` — `render`, `text`, `names`
- `templates/v2/<area>/<name>.system.j2`; `examples/v2/` optional few-shot files

## Imports

Layer 4 of 5 (`[tool.importlinter]` in the root `pyproject.toml`): may import harness_schema and harness_config only.

## Test

```sh
uv run --python 3.12 pytest -q packages/prompts
```

## Rules

- Prompt examples use made-up entities (p99, MethodX, BenchY); never text from an evaluation question.
- The template text is the version hash: runs record which text they used.

Repository-wide rules and the map: the root `AGENTS.md`.
