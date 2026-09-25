# Public deployment preparation

No Pages deployment or Tunnel is created by this change. A public hostname and Tunnel require prior approval under university/cluster policy.

## Temporary no-domain preview

For an interactive temporary preview without a custom domain, create a free Turnstile widget in Cloudflare first, then run `./scripts/public-preview.sh` from the repository root on the Slurm node. The script prompts for the site key and (hidden) secret if they are not already exported, creates a fresh local identity salt, builds the public frontend, and starts MCP, the private API, and the public gateway. The public gateway serves both the static UI and its narrow API on `127.0.0.1:8787`; private API and MCP remain on loopback. Press Ctrl-C to stop the processes.

Point a Cloudflare Quick Tunnel at **only** `http://127.0.0.1:8787`. Open the generated `trycloudflare.com` URL to use the site. Quick Tunnel URLs are temporary and change after restart. Quick Tunnels are for testing, do not provide an uptime guarantee, and do not support SSE; this UI polls run state as a fallback. No Cloudflare login or custom domain is needed for a Quick Tunnel. Do not use the development Turnstile bypass for an internet-facing preview.

## Build and launch

From repository root, install workspace dependencies, then build the public bundle:

```sh
VITE_PUBLIC_API_BASE=https://<approved-api-host> VITE_TURNSTILE_SITE_KEY=<public-site-key> corepack pnpm --dir web build:public
corepack pnpm --dir web check:public-build
```

Pages root directory: `web`; build command: `corepack pnpm build:public`; output: `dist`. Public variables are exactly `VITE_PUBLIC_INTERACTIVE`, `VITE_PUBLIC_API_BASE`, `VITE_TURNSTILE_SITE_KEY`. These are visible to browsers. Never define model identifiers, private URLs, or secrets in Vite variables.

Set these exact environment variables. Keep public build values in Pages and gateway-only values on the Slurm node:

| Variable | Classification | Purpose |
| --- | --- | --- |
| `VITE_PUBLIC_INTERACTIVE=true` | Public Pages | Selects the public entry point. |
| `VITE_PUBLIC_API_BASE=https://<approved-api-host>` | Public Pages | Public gateway origin. |
| `VITE_TURNSTILE_SITE_KEY` | Public Pages | Turnstile site key; designed to be public. |
| `PUBLIC_PRIVATE_ORIGIN=http://127.0.0.1:8000` | Gateway config | Exact loopback private API origin. |
| `PUBLIC_ALLOWED_ORIGINS` | Gateway config | Exact public site origin list. |
| `PUBLIC_TURNSTILE_ENABLED`, `PUBLIC_ENV=production`, `PUBLIC_DEV_BYPASS=false` | Gateway config | Verification and environment mode. |
| `PUBLIC_FAST_ENABLED`, `PUBLIC_ADAPTIVE_ENABLED` | Gateway config | Per-path admission switches. |
| `PUBLIC_TRUST_CLIENT_ID=false` | Gateway config | Keep false unless trusted ingress rewrites the identifier. |
| `PUBLIC_MAX_QUEUE=10`, `PUBLIC_MAX_STARTS_PER_DAY=3`, `PUBLIC_MAX_ACTIVE_PER_CLIENT=1`, `PUBLIC_MAX_IDENTITIES=10000` | Gateway config | Bounded admission and identity store. The code prevents increases above the reviewed defaults for queue and daily quota. |
| `PUBLIC_MAX_BODY_BYTES=4096`, `PUBLIC_QUESTION_MIN_CHARS=8`, `PUBLIC_QUESTION_MAX_CHARS=500` | Gateway config | Request bounds; configured values can only tighten the reviewed maxima. |
| `PUBLIC_FAST_DEADLINE_SECONDS=90`, `PUBLIC_ADAPTIVE_DEADLINE_SECONDS=180` | Gateway config | Wall-clock limits; configured values can only shorten runs. |
| `HARNESS_MAX_CONCURRENT_PUBLIC_RUNS=1` | Private API config | Public concurrency cap; one private slot is reserved. |
| `PUBLIC_IDENTITY_SALT` | Secret | Salt for client identity hashes. |
| `TURNSTILE_SECRET_KEY` | Secret | Server-side Turnstile verification only. |
| `HARNESS_MAX_CONCURRENT_RUNS` | Private API config | Existing total model run capacity; keep at least two to admit public work while reserving private capacity. |

The gateway does not load `.env`; `.env.example` documents safe names and placeholders. Never define the secret rows as Pages variables.

Launch the dedicated gateway bound to loopback only:

```sh
PUBLIC_PRIVATE_ORIGIN=http://127.0.0.1:8000 PUBLIC_ALLOWED_ORIGINS=https://<pages-host> TURNSTILE_SECRET_KEY=<secret> uv run --project packages/public_gateway uvicorn harness_public_gateway.app:app --app-dir packages/public_gateway/src --host 127.0.0.1 --port 8010
```

Configure the named Cloudflare Tunnel to target only `http://127.0.0.1:8010`. The public hostname must not target port 8000 or any model/data service. See [operations](PUBLIC_OPERATIONS.md) and [threat model](PUBLIC_THREAT_MODEL.md).

## Checklist

- Verify university/cluster approval for hostname and Tunnel.
- Configure Turnstile hostname, public site key, and gateway-only secret.
- Set exact Pages origin in gateway CORS.
- Set a private random identity salt; disable dev bypass.
- Check private API listens on loopback only and the gateway cannot be reached except through the Tunnel.
- Check public test/build output and `_headers` CSP uses the final API hostname.
- Confirm examples are empty or explicitly reviewed and exported.
- Confirm public capacity cap leaves private capacity.
- Test queue-full, quota, unavailable API and Turnstile failure responses.

## Shutdown and recovery

To stop public requests immediately, stop the gateway or stop the Tunnel ingress. Existing mappings/queue are process memory; restart invalidates them. Verify private API/model services remain private and available to researchers. Restore from a known gateway config, rotate Turnstile secret if exposed, and recheck the checklist before reopening.

For local development, use a separate gateway process/config with `PUBLIC_ENV=development` and an exact development origin such as `http://127.0.0.1:5173`. Keep that origin out of the production allowlist; never combine local bypass settings with production configuration.
