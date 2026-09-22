# Research Evaluation Context

This context defines the vocabulary for evaluating research answers and extracted knowledge graph claims.

## Evaluation language

**Atomic triple**:
A single subject–predicate–object claim that expresses one fact and can be scored independently.
_Avoid_: Compound claim, bundled triple

**Gold graph**:
The question-scoped set of canonical triples approved from the external OWL/DBLP knowledge graph and its reviewed alternatives.
_Avoid_: Ground truth corpus

**Triple equivalence**:
The rule that two triples represent the same claim after canonicalizing identifiers, predicates, aliases, and whitespace.
_Avoid_: String equality

**Interpretation disagreement**:
Divergence between independently generated readings or plans for the same question.
_Avoid_: Path disagreement

**Path disagreement**:
Divergence between the triples produced by the fast baseline, enhanced baseline, and agentic research paths for the same question.
_Avoid_: Interpretation disagreement

**Provenance coverage**:
The share of predicted triples linked to source passages or graph records that support them.
_Avoid_: Citation count
