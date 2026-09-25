#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"

if [[ -z "${VITE_TURNSTILE_SITE_KEY:-}" ]]; then
  read -r -p 'Turnstile site key (public): ' VITE_TURNSTILE_SITE_KEY
fi
if [[ -z "${TURNSTILE_SECRET_KEY:-}" ]]; then
  read -r -s -p 'Turnstile secret key (hidden): ' TURNSTILE_SECRET_KEY
  printf '\n'
fi
if [[ -z "${PUBLIC_IDENTITY_SALT:-}" ]]; then
  PUBLIC_IDENTITY_SALT=$(openssl rand -hex 32)
fi
export VITE_TURNSTILE_SITE_KEY TURNSTILE_SECRET_KEY PUBLIC_IDENTITY_SALT

if [[ "${PUBLIC_DEV_BYPASS:-false}" == "true" || "${PUBLIC_TURNSTILE_ENABLED:-true}" != "true" ]]; then
  printf 'Refusing public preview with Turnstile disabled or development bypass enabled.\n' >&2
  exit 2
fi

if [[ ! -f .env ]]; then
  printf 'Missing repository .env required by the private research API.\n' >&2
  exit 2
fi

for port in 8000 8765 8787; do
  if (echo >/dev/tcp/127.0.0.1/"$port") >/dev/null 2>&1; then
    printf 'Port %s is already in use. Stop the existing Harness process before starting this preview.\n' "$port" >&2
    exit 2
  fi
done

VITE_PUBLIC_INTERACTIVE=true \
VITE_PUBLIC_API_BASE='' \
VITE_TURNSTILE_SITE_KEY="$VITE_TURNSTILE_SITE_KEY" \
  corepack pnpm --dir web build:public

cleanup() {
  trap - EXIT INT TERM
  kill "${gateway_pid:-}" "${api_pid:-}" "${mcp_pid:-}" 2>/dev/null || true
  wait "${gateway_pid:-}" "${api_pid:-}" "${mcp_pid:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

MCP_CORPUS_TRANSPORT=streamable-http uv run --python 3.12 python -m harness_mcp_corpus.server &
mcp_pid=$!

HARNESS_MAX_CONCURRENT_PUBLIC_RUNS="${HARNESS_MAX_CONCURRENT_PUBLIC_RUNS:-1}" \
  uv run --python 3.12 uvicorn main:app --app-dir api --host 127.0.0.1 --port 8000 --env-file .env &
api_pid=$!

printf 'Waiting for the private API on loopback…\n'
ready=false
for _ in $(seq 1 60); do
  if curl --silent --show-error --fail http://127.0.0.1:8000/internal/public/capacity >/dev/null 2>&1; then
    ready=true
    break
  fi
  if ! kill -0 "$api_pid" 2>/dev/null; then
    printf 'Private API exited during startup.\n' >&2
    exit 1
  fi
  sleep 1
done
if [[ "$ready" != true ]]; then
  printf 'Private API did not become ready within 60 seconds.\n' >&2
  exit 1
fi

PUBLIC_ENV=production \
PUBLIC_DEV_BYPASS=false \
PUBLIC_TURNSTILE_ENABLED=true \
PUBLIC_PRIVATE_ORIGIN=http://127.0.0.1:8000 \
PUBLIC_ALLOWED_ORIGINS='' \
PUBLIC_IDENTITY_SALT="$PUBLIC_IDENTITY_SALT" \
PUBLIC_SERVE_FRONTEND=true \
TURNSTILE_SECRET_KEY="$TURNSTILE_SECRET_KEY" \
  uv run --project packages/public_gateway uvicorn harness_public_gateway.app:app \
    --app-dir packages/public_gateway/src --host 127.0.0.1 --port 8787 &
gateway_pid=$!

printf '\nPublic preview is ready at http://127.0.0.1:8787\n'
printf 'Point your Cloudflare Quick Tunnel at http://127.0.0.1:8787 only.\n'
printf 'Private API (8000) and MCP remain loopback-only. Press Ctrl-C to stop all services.\n\n'
wait -n "$gateway_pid" "$api_pid" "$mcp_pid"
