# Public threat model

## Boundaries and data classes

The browser can reach only Cloudflare Pages and the public gateway. Tunnel ingress must terminate at the gateway loopback listener. The gateway reaches the private Harness API at a validated loopback origin; that API reaches local model infrastructure. LiteLLM, vLLM, MCP, Postgres, Langfuse, corpus roots, artifacts and private administrative routes remain private.

Public inputs (question, path, Turnstile token) are untrusted. The gateway maps private run IDs to random public IDs and projects only supported answers, claim evidence, triples, safe graph fields, generic progress and friendly errors. Private IDs, paths, URLs, prompts, traces, stack traces, artifact contents, settings, ratings, benchmarks, gold and corpus administration are private.

## Controls

Turnstile is verified server-side and fails closed. The only bypass is explicit `PUBLIC_ENV=development`, `PUBLIC_TURNSTILE_ENABLED=false`, and `PUBLIC_DEV_BYPASS=true`; the gateway rejects bypass configuration in production. Restrict direct access to the gateway so only the Tunnel can reach it. `CF-Connecting-IP` is trusted only when the TCP peer is loopback through the local Tunnel; arbitrary forwarded headers from direct clients are ignored. A deployment-provided `X-Public-Client-Id` is ignored by default. To use it, enable `PUBLIC_TRUST_CLIENT_ID` only after trusted ingress strips/replaces the caller value; the gateway accepts it only from loopback ingress.

CORS uses exact configured origins and no credentials. Request size, question length, daily quota, per-client active count and global waiting capacity are bounded. Public model concurrency is separately capped; with total model capacity above one, one slot remains for private researchers. In-process limits are not shared across processes and reset on restart. There is no authenticated cancellation endpoint.

## Manual Cloudflare steps

Create a Turnstile widget restricted to the final Pages hostname; set its site key as the public `VITE_TURNSTILE_SITE_KEY` and its secret as gateway-only `TURNSTILE_SECRET_KEY`. Configure a Pages project from this repository with root `web`, build `corepack pnpm build`, output `dist`, and public variables `VITE_PUBLIC_INTERACTIVE=true`, `VITE_PUBLIC_API_BASE=https://<approved-api-host>`, and the site key. Create a named Cloudflare Tunnel only after university/cluster policy approves the public hostname. Route that hostname to `http://127.0.0.1:<gateway-port>` and nothing else. Do not authenticate, deploy or create the tunnel from this task.
