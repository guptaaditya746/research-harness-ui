# harness_mcp_corpus

The MCP server that exposes the corpus tools to the agent.

## Entry points

- `server.py` — `create_server`; `probe.py` — the M1 corpus probe

## Imports

Layer 3 of 5 (`[tool.importlinter]` in the root `pyproject.toml`): may import the layers below (corpus, prompts, config, ...), not its siblings.

## Test

```sh
uv run --python 3.12 pytest -q packages/mcp_corpus
```

## Rules

- A new tool also needs a decision in `harness_agent.decisions.CURATED_TOOLS`.

Repository-wide rules and the map: the root `AGENTS.md`.
