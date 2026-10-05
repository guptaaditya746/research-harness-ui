# harness_agent

The Agentic path: concept readings before research, the MCP tool loop, paper readers, the independent verifier, and the hand-off to the shared pipeline.

## Entry points

- `research.py` — `AgenticResearch.run`: the steps of one run
- `tool_loop.py` — `ToolLoop`: one agent conversation; `decisions.py` — the curated tools and the decision checks
- `mcp.py` — `MCPToolClient` (MCP sessions, phase deadline)
- `concepts.py`, `proposal.py`, `readers.py`, `verification.py`, `context.py`, `research_state.py`

## Imports

Layer 1 of 5 (`[tool.importlinter]` in the root `pyproject.toml`): may import harness_baseline and every lower layer.

## Test

```sh
uv run --python 3.12 pytest -q packages/agent
```

## Rules

- Tool names and the decision JSON are a contract with the prompts (`agent/research_supervisor`): change both together.
- The agent may only cite ids a tool returned in the session (`decision_problems`, `validate_findings`).

Repository-wide rules and the map: the root `AGENTS.md`.
