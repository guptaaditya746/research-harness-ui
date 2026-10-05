# Graph settings: a larger, more linked graph

How many triples a run extracts, how linked they are, and which settings change that. The
`exhaustive` profile turns all of them up.

**Evidence documents (D)** — one exhaustive, reviewed document per evaluation question, used as gold —
are built in labelling (`labelling evidence-docs`, labelling's `docs/EVIDENCE_DOCS.md`), apart from this
harness (decision 0003). The builder uses this harness's packages as tools and the settings below,
with the `exhaustive` profile by default.

## Where triples used to be lost, and the settings for each

| Stage | Before | Setting |
|---|---|---|
| Retrieval | 16–24 passages per run | `BASELINE_RETRIEVE_K`, `HARNESS_*_MAX_PASSAGES`, `HARNESS_RETRIEVAL_MODE` |
| Schema inference | "keep the subschema small" | `HARNESS_SCHEMA_SCOPE=broad`: the question's whole topic |
| Extraction | facts as they come | `HARNESS_GRAPH_LINKING=dense`: one name per entity, entity-to-entity relations, compound statements split into linked facts |
| Relation mapping | an open fact whose relation fits no subschema term was dropped | `HARNESS_UNMAPPED_RELATIONS=keep`: the relation is added (see below); thresholds `HARNESS_MAP_LEXICAL_MIN`, `HARNESS_MAP_MODEL_MIN` |
| Extraction size | 2 passages per call | `HARNESS_EDC_BATCH_PASSAGES` |
| Evidence review | dropped triples were gone | still dropped from the answer, but kept with their verdict in `evidence_graph.json` |

**What "unmapped" means.** Extraction first writes facts in the paper's own words
("MethodX — is fine-tuned from — ModelZ"). Each relation is then matched to a subschema term, by
spelling (`HARNESS_MAP_LEXICAL_MIN`) and then by a mapping model (`HARNESS_MAP_MODEL_MIN`). With
`drop`, a fact whose relation matches nothing is discarded although it is true and sourced. With
`keep`, the relation becomes a new property named from its wording (`isFineTunedFrom`): an OWL 2
sub-property of `researchRelation` (between things) or `researchAttribute` (a value), with domain
Paper when the subject is the paper. It joins the run's subschema, so the graph stays valid and the
scorer's entailed level can relate it to other terms through `predicate_relations.ttl`.

**Answer narrow, evidence wide.** More triples go into the evidence graph, not into the answer:
evidence review still decides what the answer uses (and what gold scoring reads), while
`evidence_graph.json` in every Adaptive and Agentic run keeps every extracted triple with
`relevant: true/false` and the reason, plus the facts lost at mapping.

## Profiles

`profiles/exhaustive.env` sets all of the above in one place (more passages, broad and dense
graphs, unmapped relations kept, one passage per extraction call, long deadlines). Use it for any run
with `HARNESS_PROFILE=exhaustive` in `.env` or the environment. Precedence: the environment, then the
profile, then `.env`, then the defaults. labelling's evidence-document builder uses it unless
`HARNESS_PROFILE` names another.
