# ADR 0005: Evaluate artifacts and control runs through the API

## Status

Accepted

## Decision

Triple metrics are computed by `harness_evaluation` from the triples in a run's
readings and a caller supplied gold set. The evaluator treats triples as unique
`(subject, predicate, object)` keys and ignores evidence fields for matching.
The API exposes that pure evaluator at `POST /runs/{id}/evaluate`.

Agentic reading selection uses `POST /runs/{id}/reading` with a `pick` index. A
stop request records a cancellation flag for an active worker; the worker writes
a terminal `stopped` run with the complete artifact set before exiting.

## Consequences

Metrics cannot be accidentally copied from model output, and offline evaluation
can reuse the same functions. Cancellation is cooperative because a Python
thread cannot be safely killed; long-running stages must return for the terminal
state to be persisted.
