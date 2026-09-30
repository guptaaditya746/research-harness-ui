#!/usr/bin/env bash
# The full harness (every page and path) on the internet, free: the UI and the API behind one
# password, through a tunnel. Only the site port goes out; the API, the corpus server and the
# models stay on loopback. See docs/PUBLIC_DEPLOYMENT.md, "Full site behind a password (free)".
#
#   SITE_PASSWORD='a long passphrase' ./scripts/public-full.sh          # Cloudflare quick tunnel (default)
#   SITE_USERS='aditya:pass-one-long,prof:pass-two-long' ./scripts/public-full.sh
#   TUNNEL=none ./scripts/public-full.sh                                # only the local site, bring your own tunnel
#
# Quick tunnel: no account and no domain, a https://<words>.trycloudflare.com address that changes on
# every start. Needs `cloudflared` on PATH (one binary, no root; see the docs).
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"
site_port="${SITE_PORT:-8080}"
tunnel="${TUNNEL:-cloudflare}"

if [[ ! -f .env ]]; then
  printf 'Missing .env (the harness settings).\n' >&2
  exit 2
fi
if [[ -z "${SITE_PASSWORD:-}" && -z "${SITE_USERS:-}" ]]; then
  read -r -s -p 'Password for the site (at least 12 characters): ' SITE_PASSWORD
  printf '\n'
fi
export SITE_PASSWORD="${SITE_PASSWORD:-}" SITE_USERS="${SITE_USERS:-}"
if [[ "$tunnel" == cloudflare ]] && ! command -v cloudflared >/dev/null 2>&1; then
  printf 'cloudflared is not installed. Install it without root:\n' >&2
  printf '  mkdir -p ~/.local/bin && curl -L -o ~/.local/bin/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x ~/.local/bin/cloudflared\n' >&2
  exit 2
fi
for port in 8000 8765 "$site_port"; do
  if (echo >/dev/tcp/127.0.0.1/"$port") >/dev/null 2>&1; then
    printf 'Port %s is already in use: stop `make demo` (or whatever holds it) first.\n' "$port" >&2
    exit 2
  fi
done

printf 'Building the UI…\n'
corepack pnpm --dir web build >/dev/null

logs="$project_dir/artifacts/public-full"
mkdir -p "$logs"
cleanup() {
  trap - EXIT INT TERM
  kill "${tunnel_pid:-}" "${site_pid:-}" "${api_pid:-}" "${mcp_pid:-}" 2>/dev/null || true
  wait "${tunnel_pid:-}" "${site_pid:-}" "${api_pid:-}" "${mcp_pid:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

MCP_CORPUS_TRANSPORT=streamable-http uv run --python 3.12 python -m harness_mcp_corpus.server >"$logs/mcp.log" 2>&1 &
mcp_pid=$!
uv run --python 3.12 uvicorn main:app --app-dir api --host 127.0.0.1 --port 8000 --env-file .env >"$logs/api.log" 2>&1 &
api_pid=$!
uv run --python 3.12 uvicorn public_site:create_app --factory --app-dir api --host 127.0.0.1 --port "$site_port" \
  --proxy-headers --forwarded-allow-ips 127.0.0.1 >"$logs/site.log" 2>&1 &
site_pid=$!

for _ in $(seq 1 90); do
  code=$(curl --silent --output /dev/null --write-out '%{http_code}' "http://127.0.0.1:$site_port/" || true)
  [[ "$code" == 401 ]] && break
  if ! kill -0 "$site_pid" 2>/dev/null; then
    printf 'The site did not start:\n' >&2
    tail -20 "$logs/site.log" >&2
    exit 1
  fi
  sleep 1
done
printf 'Site (login required) on http://127.0.0.1:%s; logs in %s\n' "$site_port" "$logs"

case "$tunnel" in
  cloudflare)
    cloudflared tunnel --no-autoupdate --url "http://127.0.0.1:$site_port" >"$logs/tunnel.log" 2>&1 &
    tunnel_pid=$!
    url=""
    for _ in $(seq 1 60); do
      url=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$logs/tunnel.log" | head -1 || true)
      [[ -n "$url" ]] && break
      sleep 1
    done
    if [[ -z "$url" ]]; then
      printf 'No tunnel address yet; see %s\n' "$logs/tunnel.log" >&2
    else
      printf '\nPublic address: %s\n(it changes on every start; share it with the login)\n' "$url"
    fi
    ;;
  none)
    printf 'No tunnel started (TUNNEL=none): point yours at http://127.0.0.1:%s only.\n' "$site_port"
    ;;
  *)
    printf 'Unknown TUNNEL=%s (cloudflare or none).\n' "$tunnel" >&2
    exit 2
    ;;
esac
printf 'Press Ctrl-C to stop everything.\n'
wait -n "$api_pid" "$site_pid" "$mcp_pid" ${tunnel_pid:+"$tunnel_pid"}
