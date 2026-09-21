# ADR 0004: Keep coverage state separate from RDF state

Status: accepted in M2.

The obligation ledger is a deterministic transition engine. It has no RDF,
model, retrieval, or API imports. Query analysis creates obligations from the
SPARQL algebra and the complete ingest fact domain. Structural obligations are
then closed from `facts.jsonl`; content obligations remain available to the
later graph builder.

The graph store owns the per-run RDF `Dataset`. Each extraction event gets one
named graph, and the graph IRI has one or more `prov:wasDerivedFrom` links to
source IDs known to the corpus. Event validation finishes before any triple is
written, so a rejected event cannot leave unprovenanced partial data.

This boundary makes ledger closure testable without RDF and makes provenance an
atomic graph-store guarantee. The orchestrator converts any `InvariantViolation`
to the terminal reason `invariant:<number>`.
