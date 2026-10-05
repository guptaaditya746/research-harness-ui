# 0007 Comparison-only modes are deleted

**Context.** Several settings kept an older behaviour for comparisons: prompt set v1 (19 frozen templates),
`HARNESS_CONTENT_CORE=none`, `HARNESS_CONCEPT_GROUNDING=legacy`, `HARNESS_AGENT_HANDOFF=ids` and
`HARNESS_AGENT_TOOLSET=all`. Every change had to keep them working.

The thesis reports triple precision and recall, and complexity, for the corpus-wide and ambiguous questions
across the agentic path, the pipeline paths and the External harness. None of those comparisons needs the old
modes.

**Decision.** Delete them. A setting that still names a deleted value fails with a message that says so.

**Consequences.** Fewer branches. The old behaviour remains reproducible from the tag `pre-refactor-2026-10-05`.
