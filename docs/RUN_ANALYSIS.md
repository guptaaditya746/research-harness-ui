# Analysing runs

Start with the runs CLI; open the artifact files only for the detail it does not show.

```sh
make runs                                   # latest 20 runs
make runs ARGS="list -n 50 --path agentic --status incomplete"
make runs ARGS="show r-1a2b3c4d"            # checks, model time by purpose, slowest calls, prompts, answer
make runs ARGS="compare r-aaaa r-bbbb"      # before/after a change
make runs ARGS="stats -n 100"               # per path: answered share, medians, top issue
make runs ARGS="list --json"                # one JSON object per run, for scripts and agents
make runs ARGS="list --fixtures"            # the recorded demo runs in web/fixtures
```

Runs live in `ARTIFACT_ROOT` (default `./artifacts`), one folder per run id.

## Reading a run, in order

1. **Status and reason** (`run.json`: `status`, `reason`, `executionStatus`, `producingPath`). `incomplete` with a reason like `deadline_exceeded`, `interrupted` or `internal_error` is a failure; `corpus_absent` and `no_evidence_found` are completed findings. An agentic run with `producingPath: baseline_fallback` was degraded: the answer came from the pipeline without the agent.
2. **Checks** (`run.json: checks`, also in `verification.json`). The first `fail` names the stage that broke: question understanding, retrieval, extraction, triple provenance, agent research (MCP), agent evidence IDs, independent verification, agentic execution.
3. **Where the time went** (`show`: model time by purpose and slowest calls). Every model call writes "Model call started" and "Model call finished · model · purpose · seconds · tokens · prompt" rows into `trace.json`. Long `schema inference` or `open extraction` calls usually mean Heavy was busy; compare with `/v1/status` of thesis-llm-setup.
4. **What the agent did** (`tool_calls.jsonl`, `research_ledger.json`, `context_checkpoints.json`, `agent_draft.md`). One line per tool call with arguments, result count, duration and status (`repeat` = answered from the ledger). `context_checkpoint` rows show when the conversation was compacted.
5. **Evidence and claims** (`retrieval.json`, `claims.json`, `readings.json`, `triples.ttl`, `subschema.ttl`). `verification.json: agent_evidence_ids` lists which agent ids were used, observed or rejected.
6. **Prompts** (`run.json: promptHashes`): template name → `set:hash`. Compare with `render(name).ref` to see whether a template changed since the run.

## Common patterns

| You see | Likely cause | Look at |
|---|---|---|
| `agentic_degraded: heavy_model_busy` | Agent runs on Heavy and Heavy is saturated | `/v1/status`, `HARNESS_HEAVY_QUEUE_THRESHOLD` |
| `agentic_degraded: agent_evidence_ids_not_observed` | Agent cited ids no tool returned | `tool_calls.jsonl`, `agent_draft.md` |
| `agent_final_not_structured` | Final agent message was not one JSON object | `agent_draft.md`; the reflection turn in the trace |
| Many `Model call failed · timed out` | Call exceeded its budget (Heavy queue or long output) | `BUDGET_*_SECONDS`, thesis-llm-setup preemptions |
| `extraction: 0 supported triples` | Passages did not support the subschema's properties | `retrieval.json`, `subschema.ttl` |
| Concept readings with 0 passages | Known issue 3 | `plans.json` (search terms) |
| `agent research (MCP)` fails with HTTP 400 "auto tool choice requires ..." | The Lite backend was started without a tool-call parser | thesis-llm-setup `tool_call_parser`, `llm-setup smoke` |
| `GET /runs/<id>` returns 404 while the run is active | Expected: a run is readable once `run.json` is written; the UI treats 404 as pending and follows the event stream | — |

See `docs/KNOWN_ISSUES.md` for problems with a known root cause.
