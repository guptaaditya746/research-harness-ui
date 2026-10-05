# mcp_corpus

The corpus MCP server the agent researches with. 33 tools over the corpus (search, grep, BM25 ranking, facts, citations, tables, aggregates, evidence readers). Also the M1 probe.

## Files

- `server.py` — `create_server`, `python -m harness_mcp_corpus.server`
- `probe.py` — corpus-only agent probe (development)

## Depends on

harness_corpus, harness_prompts, harness_config.

See `AGENTS.md` in this folder (entry points, imports, test, rules) and the root `AGENTS.md`.
