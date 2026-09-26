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

## Schema language (Phase 4)

**Subschema**:
The classes and properties one question's answer uses, declared in OWL 2 (`owl:Class`, `owl:ObjectProperty`, `owl:DatatypeProperty`) with their parents. `subschema.ttl` holds only these terms, not the whole seed ontology.
_Avoid_: Ontology (for the per-question slice)

**Term origin**:
Where a schema term or graph edge comes from: **DBLP** (a DBLP ontology term, or a harness term declared `owl:equivalentProperty` to one: title, authoredBy, year, venue), **core** (the harness seed vocabulary: Paper, usesDataset, cites, ...), or **invented** (added for one question). Shown as a badge in the Subschema panel and as the edge colour in the answer graph.
_Avoid_: New term, custom term

**Invented property**:
A property created for one question. It is always a sub-property of an existing property of the same kind (object or datatype), falling back to `hrn:researchRelation` / `hrn:researchAttribute`, and it names the relation, not its object (`uses`, not `usesRetrievalAugmentedGeneration`).
_Avoid_: Predicate per fact

**Invented class**:
A class created for one question, only when no existing class fits, always a subclass of an existing class (default `hrn:ResearchEntity`). `HARNESS_SCHEMA_NEW_CLASSES=off` forbids them.
_Avoid_: Instance class (a named method or dataset is a value, not a class)

**DBLP backbone**:
The verified corpus facts added to a research answer's graph for the papers it uses: authors, venue (one shared node per venue) and the citations among them. It is context: it appears in the answer graph and the export, not in the scored triples. Years are not graph nodes, because a shared year would link unrelated papers.
_Avoid_: Extracted fact

**Answer export**:
`triples.ttl`: the answer's extracted triples plus the DBLP facts of its papers only (query-driven), not the whole corpus.
_Avoid_: Corpus graph
