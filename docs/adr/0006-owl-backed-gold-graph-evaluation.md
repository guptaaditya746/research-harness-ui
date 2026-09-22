# Evaluate triples against an OWL-backed gold graph

**Status: accepted**

Triplet evaluation will use a question-scoped gold graph derived from the external OWL/DBLP knowledge graph, with reviewed alternatives preserved when sources conflict. The evaluator will canonicalize identifiers and predicates, split compound extractions into atomic triples, and report precision, recall, F1, provenance coverage, unsupported-triple rate, interpretation disagreement, and path disagreement. Internal `pN` storage IDs remain provenance-only; scored triples use canonical entity IRIs. This gives the three research paths a stable, auditable comparison while preserving genuine source conflicts instead of silently choosing one authority.
