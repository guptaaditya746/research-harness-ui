# harness_observability

Tracing of stages and tool calls (Langfuse when configured).

## Entry points

- `__init__.py` — `record_stage`, `record_tool_call`

## Imports

Layer 4 of 5 (`[tool.importlinter]` in the root `pyproject.toml`): may import harness_schema and harness_config only.

## Test

```sh
uv run --python 3.12 pytest -q packages/observability
```

Repository-wide rules and the map: the root `AGENTS.md`.
