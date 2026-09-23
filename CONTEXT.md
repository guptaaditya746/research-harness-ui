# Research Evaluation Context

This context defines the vocabulary for evaluating research answers and extracted knowledge graph claims.

## Evaluation language

**Atomic triple**:
A single subject–predicate–object graph edge. It is useful for graph diagnostics but may be insufficient to express a scientific claim on its own.
_Avoid_: Complete claim

**Claim subgraph**:
A connected, typed representation of one source-backed research assertion and the context needed to interpret it, such as a metric, value, comparator, benchmark, and setting.
_Avoid_: Isolated result triple

**Evidence anchor**:
A reproducible location in a source record, passage, table cell, figure annotation, or canonical metadata fact. It is scientific provenance; the tool call that retrieved it is execution provenance.
_Avoid_: Retrieval result

**Synthesis claim**:
A harness-level inference explicitly supported by source-specific claims. It is never presented as an assertion made by one source paper.
_Avoid_: Merged paper claim

**Claim context**:
The conditions under which a research claim applies, including its evaluation setting and any comparator or population needed to interpret it.
_Avoid_: Unqualified result

**Gold claim graph**:
The question-scoped set of reviewed claim subgraphs and accepted alternatives used to evaluate predictions.
_Avoid_: Ground truth corpus, gold triple list

**Triple equivalence**:
The rule that two graph edges represent the same relation after canonicalizing identifiers, predicates, aliases, and whitespace.
_Avoid_: String equality

**Corpus absence**:
A completed finding that no matching evidence exists on a named, sufficiently covered corpus evidence surface.
_Avoid_: System failure, universal nonexistence

**No evidence found**:
A completed semantic full-text search with declared but incomplete coverage. It states what the indexed procedure found, without proving corpus absence.
_Avoid_: Corpus absence, system failure

**Provisional relation**:
Query-driven local vocabulary accepted by validation and reusable through the registry, but not yet approved project ontology vocabulary.
_Avoid_: Seed ontology term

**Degraded agentic result**:
A grounded answer produced by a fallback after the requested agentic execution failed; it is an answer outcome but not a successful agentic execution.
_Avoid_: Successful agentic run

**Interpretation disagreement**:
Divergence between independently generated readings or plans for the same question.
_Avoid_: Path disagreement

**Path disagreement**:
Divergence between the triples produced by the fast baseline, enhanced baseline, and agentic research paths for the same question.
_Avoid_: Interpretation disagreement

**Provenance coverage**:
The share of predicted claims linked to source passages or graph records that support them.
_Avoid_: Citation count
