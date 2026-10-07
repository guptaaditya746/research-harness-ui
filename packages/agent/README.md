# agent

Agentic research path. Before research, passages are retrieved and the agent proposes concept readings grounded in them; the run pauses for the user's pick. An MCP tool-using agent (Lite) then researches the corpus and records findings, parallel paper readers add more, and the findings with their evidence seed the shared pipeline (Heavy); an independent verifier checks the cited evidence.

## Files

- `research.py` — `AgenticResearch.run`: the named steps of one run
- `proposal.py`, `concepts.py` — concept readings (retrieve first, hide ungrounded ones)
- `mcp.py` — `MCPToolClient`: MCP sessions, single and batched tool calls, the agent phase deadline
- `tool_loop.py` — `ToolLoop`: one agent conversation (tool budget as a hard cap per call, turn cap = budget + 8 with a last decision-only turn, forced first call, summary compaction, decision sent back once)
- `decisions.py` — the curated tools, `record_finding`, and the checks of findings and of the final decision
- `readers.py` — parallel paper readers (`HARNESS_SUBAGENT_READERS`)
- `context.py`, `research_state.py` — context-window fitting (observation masking, token calibration), research ledger and findings
- `verification.py` — independent verifier: a fresh context and, by default, a different model (`HARNESS_MODEL_JUDGE`); it sees the agent's reasoning summary with its finding statements and the text of every cited id found in the session's tool results (reader passages, triples, `*_ids` list entries included) or the preliminary claims. It runs on the agent's own model when the judge is that model or is deferred under heavy load (agent on `HARNESS_MODEL_HIGH`, queue at `HARNESS_HEAVY_QUEUE_THRESHOLD`); see docs/ARCHITECTURE.md
- `doc_research.py`, `doc_tools.py` — the evidence document D as input: D-only tools (`doc_search` and `doc_triples` page with `offset`, report `total`, `showing` and `next_offset`)
- `load.py` — Heavy load from the status API
- `skills.py` — research playbooks loaded into the supervisor prompt

## Depends on

harness_baseline, harness_extraction, harness_prompts, harness_config, MCP server over HTTP.

See `AGENTS.md` in this folder (entry points, imports, test, rules) and the root `AGENTS.md`.
