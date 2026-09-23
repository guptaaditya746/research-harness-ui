#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"

cleanup() {
  trap - EXIT INT TERM
  kill "${api_pid:-}" "${web_pid:-}" "${mcp_pid:-}" 2>/dev/null || true
  wait "${api_pid:-}" "${web_pid:-}" "${mcp_pid:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# The agentic path researches through the MCP corpus server; start it first.
MCP_CORPUS_TRANSPORT=streamable-http uv run --python 3.12 python -m harness_mcp_corpus.server &
mcp_pid=$!

uv run --python 3.12 uvicorn main:app --app-dir api --host 127.0.0.1 --port 8000 &
api_pid=$!

(cd web && exec corepack pnpm dev) &
web_pid=$!

printf '\nResearch Harness: http://127.0.0.1:25173 on the laptop\n'
printf 'Press Ctrl-C to stop the MCP server, API and UI.\n\n'

wait -n "$api_pid" "$web_pid" "$mcp_pid"
