# 0003 The evidence-document builder moves to labelling

**Context.** An evidence document (D) is gold for its question, and labelling is where gold is built and
reviewed. The builder uses this harness only as a tool (readers, concept proposal, extraction).

**Decision.** The builder (`harness_agent.evidence_doc`) moves to labelling and calls the harness packages from
a pinned checkout of this repository.

**Consequences.** The system under test and the instrument that measures it live apart, which is easier to
defend. The harness keeps everything a run needs, including `evidence_graph.json` and the graph settings
(`HARNESS_SCHEMA_SCOPE`, `HARNESS_GRAPH_LINKING`, `HARNESS_UNMAPPED_RELATIONS`, profiles).
