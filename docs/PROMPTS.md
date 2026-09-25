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

## The 16 prompts

| Name | Model call |
|---|---|
| `planning/query_plan` | Question understanding: typed query plan |
| `agent/concept_proposal` | 3–4 concept readings before agentic research |
| `agent/research_supervisor` | System prompt of the agent tool loop (skills rendered in) |
| `agent/decision_rejected` | Message sending an invalid agent decision back with its problems |
| `agent/grounded_context` | Research context message (JSON appended) |
| `agent/independent_verifier` | Fresh-context check of the agent's cited evidence |
| `extraction/schema_inference` | Subschema for the question |
| `extraction/triple_extraction` | Triple extraction in non-EDC mode (unused by live runs, see KNOWN_ISSUES 7) |
| `extraction/edc_extract` | Open propositions from passages |
| `extraction/edc_define` | Definitions of open relations |
| `extraction/edc_map` | Mapping open relations onto ontology properties |
| `extraction/evidence_review` | Keep only supported, relevant triples |
| `extraction/answer_composition` | Answer and SPARQL from verified triples |
| `extraction/answer_verification` | Remove unsupported claims from the draft answer |
| `extraction/claim_enrichment` | Missing context fields of result claims, with exact quotes |
| `mcp_corpus/probe` | M1 corpus-only probe (development tool) |

## Sets

- **v1** is the text the harness sent before templating, reproduced byte for byte and checked against `packages/prompts/tests/golden/`. Never edit it; it exists so runs can be compared.
- **v2** follows current guidance for non-thinking instruct models (Anthropic prompt-engineering docs, OpenAI GPT-4.1/GPT-5 prompting guides, Qwen3-2507 and Granite 4.x documentation): role and task first; inputs described; numbered steps where order matters; rules with the reason behind them, stated positively; explicit output schema with an example using real ontology terms; what to return when evidence is missing; evidence quoted before the claim; documents before the question.

## Changing a prompt

1. Edit the v2 template (or add a v3 set for a larger experiment: copy `v2/`, add the set name to `SETS`).
2. Keep the output keys the parser reads (see the call site in `engine.py`, `planner.py`, `concepts.py`, `mcp.py`, `verification.py`). Schemas with `extra="forbid"` (`SchemaInference`, `ExtractedTriple`, `AnswerResult`) reject any additional key.
3. `make check` renders every template (missing variables fail).
4. Compare on the gold set (`HARNESS_PROMPT_SET=v1` vs `v2`) before relying on a change; runs record which version they used.
