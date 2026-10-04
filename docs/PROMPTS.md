# Prompts

Every model call renders a named Jinja template; no prompt text lives in Python.

```python
from harness_extraction import complete
from harness_prompts import render

prompt = render("extraction/edc_define", {"relations": relations})
reply = complete(model, prompt)          # records prompt.ref (name@set:hash) in the model events
```

- Templates: `packages/prompts/src/harness_prompts/templates/<set>/<area>/<name>.system.j2` (plus an optional `.user.j2` for prose user messages).
- Payloads are sent as JSON. In v2 long parts (passages, candidates, triples, claims) come first and the question last.
- `HARNESS_PROMPT_SET` selects the set: `v2` (default) or `v1`.
- `Run.prompt_hashes` records the versions a run used.

## The 19 prompts

| Name | Model call |
|---|---|
| `planning/query_plan` | Question understanding: typed query plan |
| `agent/concept_proposal` | 3–4 concept readings before agentic research |
| `agent/research_supervisor` | System prompt of the agent tool loop (skills rendered in) |
| `agent/decision_rejected` | Message sending an invalid agent decision back with its problems |
| `agent/grounded_context` | Research context message (JSON appended) |
| `agent/independent_verifier` | Fresh-context check of the agent's cited evidence |
| `agent/compaction_summary` | Summary of the oldest tool exchanges when the agent's context is 80% full (Phase 3c; v1 = v2) |
| `agent/paper_reader` | Parallel reader: up to four findings from one paper's top passages (Phase 3d; v1 = v2) |
| `extraction/schema_inference` | Subschema for the question |
| `extraction/triple_extraction` | Triple extraction in non-EDC mode (unused by live runs, see KNOWN_ISSUES 7) |
| `extraction/edc_extract` | Open propositions from passages |
| `extraction/edc_define` | Definitions of open relations |
| `extraction/edc_map` | Mapping open relations onto ontology properties |
| `extraction/evidence_review` | Keep only supported, relevant triples |
| `extraction/answer_composition` | Answer from verified triples (v2: no SPARQL; the query is built from the cited triples) |
| `extraction/answer_verification` | Remove unsupported claims from the draft answer |
| `extraction/claim_enrichment` | Missing context fields of result claims, with exact quotes |
| `mcp_corpus/probe` | M1 corpus-only probe (development tool) |
| `evaluation/triple_match` | Soft-match judge: does a run triple state the same fact as a gold triple? (added after templating; v1 = v2) |

## Sets

- **v1** is the text the harness sent before templating, reproduced byte for byte and checked against `packages/prompts/tests/golden/`. Never edit it; it exists so runs can be compared.
- **v2** follows current guidance for non-thinking instruct models (Anthropic prompt-engineering docs, OpenAI GPT-4.1/GPT-5 prompting guides, Qwen3-2507 and Granite 4.x documentation): role and task first; inputs described; numbered steps where order matters; rules with the reason behind them, stated positively; explicit output schema with an example using real ontology terms; what to return when evidence is missing; evidence quoted before the claim; documents before the question.

## Graph shape variables

`schema_inference`, `edc_extract` and `triple_extraction` (v2) take `schema_scope`
(`focused`/`broad`) and `graph_linking` (`standard`/`dense`) from `HARNESS_SCHEMA_SCOPE` and
`HARNESS_GRAPH_LINKING`. With the defaults they render the same text as before; `broad` replaces
"keep the subschema small" with covering the question's topic, `dense` asks for one name per
entity, relations between research entities, and compound statements split into linked facts.
Their examples use made-up entities only (MethodX, ModelZ, BenchY). See `docs/EVIDENCE_DOCS.md`.

## Changing a prompt

1. Edit the v2 template (or add a v3 set for a larger experiment: copy `v2/`, add the set name to `SETS`).
2. Keep the output keys the parser reads (see the call site in `engine.py`, `planner.py`, `concepts.py`, `mcp.py`, `verification.py`). Schemas with `extra="forbid"` (`SchemaInference`, `ExtractedTriple`, `AnswerResult`) reject any additional key.
3. `make check` renders every template (missing variables fail).
4. Compare on the gold set before relying on a change; runs record which version they used.

## Few-shot examples (Phase 3f)

Five v2 prompts show worked examples when a file for them exists: `extraction/edc_extract`,
`extraction/evidence_review`, `extraction/answer_composition`, `agent/paper_reader` and
`agent/research_supervisor`. The file is `packages/prompts/src/harness_prompts/examples/v2/<area>/<name>.json`,
a list of `{"input": ..., "output": ..., "why": ...}`. The template renders them under "Examples from
earlier errors" just before the output instructions. The file's text is part of the prompt hash, so a
run's `prompt_hashes` tell which examples it saw. No example files ship: the examples should come from
this harness's own errors, not invented ones.

Choosing them:

1. Split the gold questions once, before looking at errors: a development part (about a third, spread
   over simple/compound/complex) and a test part. Write the development ids to a file and never change
   the split afterwards. Report results only on the test part; examples taken from test questions would
   inflate the scores (leakage).
2. Run the paths on the development questions, then `make fewshot ARGS="--dev @dev_ids.txt"`. It lists
   missed vital gold triples (edc_extract candidates: the passage that states the fact and the
   proposition that should have come out) and extras reviewers rejected as wrong (evidence_review
   candidates: the candidate with its evidence and `{"keep_ids": []}`), repeated errors first, and
   writes the full list to `results/fewshot_candidates.json`.
3. Pick two to four per prompt that show different error types (a result without its benchmark, a
   reversed relation, an overstated claim). Shorten the input to what the example needs, write the
   "why" in one sentence (the test rejects a leftover TODO), and save the file.
4. `make check`, then compare with and without the file on the test questions. Templates and examples are read once per process: restart the API and workers after adding, editing or removing a file, and check that the prompt hash in the runs changed. Keep an example only if
   recall rises without precision dropping. Short examples matter at 32k context: each one is sent with
   every call of that prompt.

The `answer_composition`, `paper_reader` and `research_supervisor` examples have no generator; write
them from failed runs you have inspected (`make runs ARGS="show <id>"`).
