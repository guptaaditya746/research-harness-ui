# ADR 0007: Build a query-driven graph of provenance-bearing claim subgraphs

Status: accepted. Supersedes ADR 0006 as the primary semantic and evaluation target; atomic edge metrics remain diagnostics.

Scientific results need their metric, value, unit, comparator, benchmark, setting, and source to remain connected. The harness will construct and reuse typed claim subgraphs from question-relevant evidence, attach provenance to each claim, and generate factual prose from validated claims. DBLP/OWL supplies seed vocabulary, while evidence-backed local relations can express concepts the seed cannot represent faithfully. This replaces isolated triples as the main unit of extraction, reasoning, and gold evaluation without requiring a universal ontology or full nanopublication framework.

Production rules operate on question and evidence classes, not known paper titles, values, or expected answers. The initial claim shapes are ResultClaim, ComparisonClaim, MethodClaim, and LimitationClaim; direct claim relations include supports, contradicts, qualifies, sameContext, differentContext, and possibleConflict. Trusted structured metadata can be treated as canonical metadata, while scientific results remain attributed assertions.

The user chooses Fast, Adaptive, Agentic, or Compare All before research. Verified corpus absence is a completed outcome with a declared evidence surface and coverage; failed research remains incomplete. A grounded fallback after agentic failure remains visible but is excluded from successful agentic scoring. Agentic research stops when answer requirements are evidenced or the relevant absence procedure is exhausted.

The system exposes plausible interpretation options, including “Combine all applicable interpretations.” It does not infer ambiguity from keywords alone. Malformed values and ontology type mismatches are rejected; a supported claim can instead use a suitable local relation. Every externally verifiable statement in prose must be recoverable from an accepted claim subgraph and its evidence.

Claim alignment, identity, and claim-subgraph scoring remain open design decisions. Until those are settled, existing triple metrics describe graph edges and do not constitute claim-level precision or recall.
