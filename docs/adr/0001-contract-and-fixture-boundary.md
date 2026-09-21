# ADR 0001: Extend the handoff run contract for terminal state metadata

Status: accepted for M0.

The handoff JSON is the UI contract. The v2 plan also requires `stopped`,
`reason`, coverage, threshold, model IDs, prompt hashes, and an artifact index.
The Pydantic model keeps every handoff field and adds these fields. Generated
TypeScript and fixtures come from this model in one `make fixtures` step.

The UI does not derive scientific metrics. In M0, fixtures carry clearly
synthetic values. In later milestones `packages/evaluation` will compute them
from saved artifacts before a run is returned by the API.
