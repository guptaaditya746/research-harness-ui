# agent

Agentic research path. An MCP tool-using agent (Lite) researches the corpus; its evidence seeds the shared pipeline (Heavy); an independent verifier checks the cited evidence. Before research the agent proposes concept readings and the run pauses for the user's pick.

## Files

- `research.py` — `AgenticResearch.run`: the named steps of one run
- `proposal.py`, `concepts.py` — concept readings
- `mcp.py` — `MCPToolClient`, the agent tool loop, decision validation
- `context.py`, `research_state.py` — context-window fitting, research ledger
- `verification.py` — independent verifier
- `load.py` — Heavy load from the status API
- `skills.py` — research playbooks loaded into the supervisor prompt

## Depends on

harness_baseline, harness_extraction, harness_prompts, harness_config, MCP server over HTTP.

See `AGENTS.md` for the repository map and conventions.
