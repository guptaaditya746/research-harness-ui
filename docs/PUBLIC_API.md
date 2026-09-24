# Public API

The gateway exposes only these routes. Unknown routes are not proxied.

- `POST /v1/public/runs`: `{question, path, turnstileToken}`; accepts `fast` or `adaptive`, returns opaque `publicRunId`, state, and queue position.
- `GET /v1/public/runs/{public_run_id}`: question, selected path, safe state/progress, supported answer, projected evidence/triples/graph, and friendly failure.
- `GET /v1/public/runs/{public_run_id}/events`: SSE containing opaque ID, safe state/progress, and answer only.
- `GET /v1/public/status`: aggregate public queue availability and enabled paths.
- `GET /v1/public/examples`: curated static examples, empty until reviewed IDs are allowlisted.

Public states are queued, running, awaiting_reading, answered, incomplete, stopped, rejected. Private run status/reason/trace/artifact data are never serialized. Unsupported answers are omitted. Agentic is disabled by default.

The private API also has two loopback-only gateway seams: `GET /internal/public/capacity` returns only API availability and whether both public/total slots can be acquired; `POST /internal/public/reservations` atomically reserves one public and total slot for 15 seconds. The gateway sends that one-use reservation in a private `X-Public-Reservation` header when posting `/runs`. These routes are on the private API only and are never proxied by the public gateway.
