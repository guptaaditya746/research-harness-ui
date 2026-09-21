#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"

cleanup() {
  trap - EXIT INT TERM
  kill "${api_pid:-}" "${web_pid:-}" 2>/dev/null || true
  wait "${api_pid:-}" "${web_pid:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

uv run --python 3.12 uvicorn main:app --app-dir api --host 127.0.0.1 --port 8000 &
api_pid=$!

(cd web && exec corepack pnpm dev) &
web_pid=$!

printf '\nResearch Harness: http://127.0.0.1:25173 on the laptop\n'
printf 'Press Ctrl-C to stop the API and UI.\n\n'

wait -n "$api_pid" "$web_pid"
