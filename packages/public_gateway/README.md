# public_gateway

The narrow internet-facing gateway for public demo runs. Rate limits, Turnstile, capacity reservations; talks to the API over HTTP only (an import contract forbids harness internals).

## Files

- `app.py` — the gateway app and its configuration (`PUBLIC_*` variables)

## Depends on

nothing from the harness.

See `AGENTS.md` for the repository map and conventions.
