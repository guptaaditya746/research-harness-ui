# Analysing runs

Start with the runs CLI; open the artifact files only for the detail it does not show.

```sh
make runs                                   # latest 20 runs
make runs ARGS="list -n 50 --path agentic --status incomplete"
make runs ARGS="show r-1a2b3c4d"            # checks, model time by purpose, slowest calls, prompts, answer
make runs ARGS="compare r-aaaa r-bbbb"      # before/after a change
make runs ARGS="stats -n 100"               # per path: answered share, medians, top issue
make runs ARGS="failures -n 200"            # incomplete runs grouped by reason, with the failing check's detail
make runs ARGS="list --json"                # one JSON object per run, for scripts and agents
make runs ARGS="list --fixtures"            # the recorded demo runs in web/fixtures
make graph-shape ARGS="-n 100"              # answer-graph shape: DBLP/invented mix, connectivity, dead ends
make run-questions                         # the 30 evaluation questions through the API (Agentic: every reading)
```

Runs live in `ARTIFACT_ROOT` (default `./artifacts`), one folder per run id.

## Reading a run, in order

1. **Status and reason** (`run.json`: `status`, `reason`, `executionStatus`, `producingPath`). `incomplete` with a reason like `deadline_exceeded`, `interrupted` or `internal_error` is a failure; `answer_unsupported` means the composed answer was not backed by the extracted triples (usually the verifier removed every claim; the check says how many claims and triples; runs before 27 Sep 2026 reported this as `schema_rejected`); `corpus_absent` and `no_evidence_found` are completed findings. An agentic run with `producingPath: baseline_fallback` was degraded: the answer came from the pipeline without the agent.
2. **Checks** (`run.json: checks`, also in `verification.json`). Phase 5 added: `answer composition` warn (the first answer was repaired once; no sentence passed, or none cited a triple's source, and the answer states the extracted facts; or sentences citing ids that are no triple's source, such as `research_notes`, were dropped), `schema` (OWL 2 typing, parents, SHACL, export parses), and on Agentic `agent research (MCP)` skipped when the reading is a fact lookup ("Routing the reading" in the trace). `readings[].answerSentences` lists every sentence the composer wrote with the verifier's verdict. The first `fail` names the stage that broke: question understanding, retrieval, extraction, triple provenance, agent research (MCP), agent evidence IDs, independent verification, agentic execution.
3. **Where the time went** (`show`: model time by purpose and slowest calls). Every model call writes "Model call started" and "Model call finished · model · purpose · seconds · tokens · prompt" rows into `trace.json`. Long `schema inference` or `open extraction` calls usually mean Heavy was busy; compare with `/v1/status` of thesis-llm-setup.
4. **What the agent did** (`tool_calls.jsonl`, `research_ledger.json`, `context_checkpoints.json`, `agent_draft.md`). One line per tool call with arguments, result count, duration and status (`repeat` = answered from the ledger; `unavailable` = a tool not offered that turn, because the tool budget was used up or it is outside the curated set, refused without calling MCP). `context_checkpoint` rows show when the conversation was compacted (summary or ledger note). `record_finding` rows are the agent's findings; `paper_reader` rows are the parallel readers (paper, passages shown, findings). `research_ledger.json: findings` and `verification.json` list the findings that were handed to the pipeline; the "agent passage seeds" check says how many.
5. **Evidence and claims** (`retrieval.json`, `claims.json`, `readings.json`, `triples.ttl`, `subschema.ttl`). `retrieval.json: retrieval_mode` is `lexical` or `hybrid`; in hybrid mode the extra round's `retrieval` says whether dense embeddings were used (`bm25+dense`) or not (`bm25`). `verification.json: agent_evidence_ids` lists which agent ids were used, observed or rejected.
6. **Schema and graph shape** (`run.json: readings[].schemaTerms`, `subschema.ttl`, `make graph-shape ARGS="--run <id>"`). Each term has an origin (DBLP / core / invented) and a parent; fact edges in `readings[].graph` carry the same origin. Fact edges with ids `b-...` are the DBLP backbone (added, not extracted). In `graph-shape`, `Ext` counts only the extracted facts (Inv and Stmt are shares of those), and `Funnel P>O>M>K` follows the extraction batches: passages read > open propositions > mapped onto the schema > kept by evidence review, so a thin answer shows where it lost its facts. Evidence review records why it dropped each triple (`extraction_batches.json`, entry `evidence_review`: `dropped` counts and `dropped_examples`); `graph-shape` prints the counts, and examples with `--run`. A run whose extracted triples all fail graph building (e.g. a paper-valued relation naming a paper outside the corpus) ends `no_results` and says so in the `triple provenance` check, instead of an answer with 0 facts.
7. **Prompts** (`run.json: promptHashes`): template name → `set:hash`. Compare with `render(name).ref` to see whether a template changed since the run.

## Common patterns

| You see | Likely cause | Look at |
|---|---|---|
| `agentic_degraded: heavy_model_busy` | Agent runs on Heavy and Heavy is saturated | `/v1/status`, `HARNESS_HEAVY_QUEUE_THRESHOLD` |
| `agentic_degraded: agent_evidence_ids_not_observed` | Agent cited ids no tool returned | `tool_calls.jsonl`, `agent_draft.md` |
| `agent_final_not_structured` | Final agent message was not one JSON object | `agent_draft.md`; the reflection turn in the trace |
| Many `Model call failed · timed out` | Call exceeded its budget (Heavy queue or long output) | `BUDGET_*_SECONDS`, thesis-llm-setup preemptions |
| `extraction: 0 supported triples` | Passages did not support the subschema's properties | `retrieval.json`, `subschema.ttl` |
| Concept readings with 0 passages | `HARNESS_CONCEPT_GROUNDING=legacy`, `HARNESS_HIDE_UNGROUNDED_READINGS=false`, or no reading had support (then none is hidden) | the proposal run's `disagreement.json` (`hidden_ungrounded`, `evidence_shown`) |
| Fewer readings than usual | Ungrounded readings were hidden (Phase 3a) | the "Readings hidden" trace row |
| Agent stops researching early and decides | Tool budget reached (`HARNESS_AGENT_MAX_TOOL_CALLS`) | count of corpus calls in `tool_calls.jsonl` |
| `agent research (MCP)` fails with HTTP 400 "auto tool choice requires ..." | The Lite backend was started without a tool-call parser | thesis-llm-setup `tool_call_parser`, `llm-setup smoke` |
| `GET /runs/<id>` returns 404 while the run is active | Expected: a run is readable once `run.json` is written; the UI treats 404 as pending and follows the event stream | — |

See `docs/KNOWN_ISSUES.md` for problems with a known root cause.
