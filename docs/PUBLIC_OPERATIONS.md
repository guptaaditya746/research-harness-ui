# Public operations

The gateway limiter and mapping store are in process. A restart drops IDs and queued work; clients receive unavailable/not found and must submit again. It is single-node only and can be bypassed by horizontal replicas. Move to a durable store only when multi-process deployment, restart survival, or sustained queue volume is required; retain atomic quota, expiry, opaque mapping and capacity semantics.

Stop acceptance immediately by stopping the gateway process or setting its acceptance control in a future managed deployment; currently stopping its loopback listener is the immediate control. Disable Fast or Adaptive with `PUBLIC_FAST_ENABLED=false` or `PUBLIC_ADAPTIVE_ENABLED=false`. Agentic is hard-disabled until a public reading-choice contract is implemented. Rotate Turnstile by creating a replacement secret in Cloudflare, updating `TURNSTILE_SECRET_KEY` in the service environment, restarting the gateway, then revoking the old secret. Never place the secret in Pages variables.

For abuse investigation, use aggregate queue/status and gateway service logs. Do not log questions, tokens, IPs, identity hashes, prompts, model responses or traces. Keep only incident-specific minimal timestamps and opaque public run IDs, then expire them. Do not enable verbose HTTP body logging.

Default limits are one active run per client, three starts per client per rolling 24 hours, ten waiting requests globally, 90 seconds for Fast, 180 seconds for Adaptive, a 500 character maximum question, and a 4 KiB request body. The identity record store is capped at 10,000 clients. Queue/daily/deadline/body limits can be tightened; the code prevents raising the reviewed queue, daily quota, question length, body size, and path deadline maxima.
