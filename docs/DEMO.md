# Live demo runbook

## 1. Before the demo (about 10 minutes)

On the Slurm compute node, inside the allocation:

```sh
git pull
uv sync --python 3.12 --group dev
corepack pnpm install
```

Check `.env` against `.env.example`. These settings matter for a live session:

| Variable | Demo value | Why |
|---|---|---|
| `HARNESS_INCLUDE_FIXTURES` | `false` | hides the four recorded runs from Runs and Recent |
| `BUDGET_BASELINE_FAST_SECONDS` / `_SLOW_SECONDS` | `60` / `90` | per model call; Qwen structured calls often exceed 15 s |
| `HARNESS_MCP_AGENT_TIMEOUT_S` | `90` | time available for MCP research; observed passages can still seed retrieval if the controller times out |
| `HARNESS_RUN_DEADLINE_FAST_S` / `_SLOW_S` | `300` / `900` | hard limit per run |
| `HARNESS_RUN_DEADLINE_AGENTIC_SLOW_S` | `300` | agentic presentation run ends within five minutes |
| `HARNESS_ADAPTIVE_MAX_PASSAGES` / `HARNESS_AGENTIC_MAX_PASSAGES` | `16` / `24` | bounds the number of open extraction batches |
| `HARNESS_EXTRACTION_CONCURRENCY` | `3` | extraction and evidence-review batches sent to the model at once |
| `HARNESS_AGENTIC_MAX_MCP_SEEDS` | `12` | limits MCP passages entering extraction and heavy-model verification |
| `HARNESS_AGENT_CONTEXT_TOKENS` | `32768` | the served Granite `max_model_len` in thesis-llm-setup; the agent's conversation is fitted to this window. A larger value makes the first long turn fail before the harness learns the real limit |
| `LLM_STATUS_URL` | `http://127.0.0.1:8010/v1/status` | thesis-llm-setup status API; the harness defers heavy calls when Heavy is saturated |

**Models.** Each run makes structured calls through `HARNESS_MODEL_SMALL` and
`HARNESS_MODEL_EDC` (open extraction, predicate definition, and relation mapping).
`HARNESS_MODEL_EDC` defaults to `HARNESS_MODEL_SMALL`; set it to a smaller instruct
alias when broad extraction would create too many slow calls. Schema inference and
evidence review remain on `HARNESS_MODEL_HIGH`, and agentic concepts and MCP control
use `HARNESS_MODEL_AGENT`. Point these roles at instruct models. A thinking model
reasons for hundreds of tokens before every JSON reply, so runs take minutes and time out.
If a thinking model has to be used, turn its thinking off for these calls with
`HARNESS_MODEL_OPTIONS` (see `.env.example`). `make preflight` warns when a model reasons
before answering.

Start the laptop tunnel and the stack:

```sh
omen-tunnels laptop start all        # on the laptop
omen-tunnels compute start all       # on the compute node
make demo                            # MCP corpus server + API + UI, without API auto-reload
```

Use `make dev` while editing code; it starts Uvicorn with auto-reload. For the
presentation, stop that stack and use `make demo` so a file change cannot
restart the API in the middle of a run. The UI still updates through Vite.

In a second shell on the node, run both checks. Fix every `FAIL` before you present:

```sh
make preflight       # config, corpus, each model alias, Granite tool calling, MCP, API, storage
make eval-planner    # question understanding on 29 phrasings; should be >= 85 %
```

If `eval-planner` misclassifies a question type, write it down: questions of that type will
fall back to full-text research. They still get an answer, only a slower one.

Open `http://127.0.0.1:25173` on the laptop.

## 2. What to show

No typing is needed. The start screen lists the 60 evaluation questions (nlp01-nlp30 and nlp32-nlp61,
the same list as **Question library** under the question box). Hover a question to run it with **Fast**,
**Adaptive**, **Agentic** or **Compare**; click its text to put it in the box and edit it first.
Filter by any word. The list comes from `HARNESS_QUESTION_SETS` (default
`question_library_2025.json,question_library_2025_b.json`); set `HARNESS_QUESTION_EXTRAS=true` to also list gold candidates and
recent runs.

The corpus has 337 DBLP-verified papers: the original 37, 150 ACL 2025 long
papers, and 150 EMNLP 2025 main papers. The evaluation questions are
`datasets/question_library_2025.json` (nlp01-nlp30, mostly one paper each) and
`datasets/question_library_2025_b.json` (nlp32-nlp61: corpus-wide questions over many papers,
professor-style ambiguous ones, and more questions answered from DBLP facts; check them with
`uv run python scripts/question_coverage.py` before building gold); `datasets/question_matrix_20.json` and `corpus_10.json` stay
available through `HARNESS_QUESTION_SETS`. Several rows of the table below (the `q..` questions) come
from `question_matrix_20.json`: add it to `HARNESS_QUESTION_SETS` to list them, or type them.

| Show | Path | Question | Expected |
|---|---|---|---|
| Question understanding | Fast | *Who authored FOCUS: object-centric world models for robotic manipulation?* | answered from verified author facts; trace starts with `Understanding question · paper_authors` |
| Same meaning, different wording | Fast | *List the people who wrote FOCUS.* | same answer and same trace, because the plan routes the question, not the wording |
| New ACL authors | Fast | *List the authors of FACT-AUDIT.* | answered from verified author facts and official 2025 ACL metadata |
| New EMNLP authors | Fast | *List the authors of CondAmbigQA.* | answered from verified author facts and official 2025 EMNLP metadata |
| Full-text research | Adaptive | *What improvement did Video Prediction Policy report on CALVIN ABC→D?* | 41.5 % result, cited to a passage, with the claim subgraph |
| Agent concepts | Agentic | *Who is the best author in the corpus?* | the agent proposes 3–4 interpretations with corpus evidence counts; pick one |
| Agentic research | Agentic | *What limitation of learning on physical robots does the DayDreamer paper report?* | pick a concept → MCP tool calls in the trace → verified answer |
| Side by side | Compare all | *Which paper introduced Video Prediction Policy?* | fast and adaptive finish; then you choose the agentic concept |

What every answer screen shows, top to bottom:

- **Answered using: …** — the measure the answer is computed with (for example *number of
  distinct corpus papers per author* versus *number of distinct corpus papers citing each
  author's papers*). An ambiguous question such as "best author" is answered under one
  stated measure, never silently under another.
- **Answer graph** — one connected graph, not a list of triples: each paper, author or value
  is one node, so three papers by the same author meet at one author node. Solid edges are
  source facts (click one to see its source id and evidence), dotted edges lead to evidence
  passages, and dashed boxes are results the harness *computed* (such as "3 distinct
  papers"); clicking one highlights exactly the facts it was computed from. Result claims
  connect a number to its metric, benchmark, paper and passage.
- **Checks** — one line per stage. In Agentic mode, *agent research (MCP)* runs after
  concept selection and supplies passage IDs before retrieval, extraction and answer
  verification. The checks also show *agent evidence IDs* and *agentic execution*.
  "MCP found no matches", "extraction produced no supported triples"
  and "the agent's evidence IDs were never returned by a tool" therefore read differently.

How the agent is kept honest and bounded (say this if asked about "agentic behaviour"):

- **Ledger, not truncation.** When the agent's conversation outgrows the model window, the
  oldest tool exchanges are replaced by a digest of what every call asked and found (ids and
  one line per item), so nothing gathered is lost. The trace shows a *Context checkpoint*.
- **Loop guards.** An identical tool call is answered from the ledger without going to the
  server (*repeated call answered from the ledger* in the trace). After four per-paper calls
  of one tool the result carries a hint pointing at `aggregate_facts` / `metadata_table`.
- **One self-correction.** A final decision that cites ids no tool returned, names unknown
  claim ids, is not JSON, or was made without any tool call is sent back once with exactly
  those problems (*Agent decision sent back*). Only if the corrected decision still fails
  does the run degrade.
- **Independent verifier.** A separate structured call in a fresh context (the judge model)
  sees only the question, the decision and the text of the evidence it cites, and marks each
  cited item supported or not. It is advisory: the *independent verification* check reads
  pass/warn, it never changes the answer.

There is deliberately no multi-agent debate: on a 30B/8B setup it multiplies latency and
context pressure without a measurable gain.

While a run is in progress the screen shows the current stage, the model call in flight, the
number of tool calls and the last tool result, the passages and papers examined, and the
elapsed time. For corpus-wide questions the agent has `aggregate_facts` and
`metadata_table`, so counting an author's papers is one call that cites its facts rather
than one `structured_facts` call per paper.

Gold and benchmark flow:

1. On an answered result, rate it in the sidebar, then **Promote to gold**.
2. On **Gold**, open the candidate, review the claims JSON and press **Save and approve**.
   Invalid JSON or a malformed claim is refused with the exact field.
3. On **Benchmark**, press **Run benchmark · 3× per path** for that question, then
   **Refresh** once the runs finish. Agentic repetitions research the agent's
   top-ranked concept and record it as automatically selected.

## 3. If something goes wrong

| Symptom | Meaning | Action |
|---|---|---|
| A run seems stuck | the trace shows "Waiting Ns · <model> · <stage>" for the model call in flight | if one model stage keeps taking long, that model is the bottleneck; see the model note in section 1 |
| Failed: "A model call failed: timed out after …" | that model call exceeded its per-call timeout | a thinking model on structured calls is the usual cause (section 1) |
| "Queued for model capacity" in the trace | another agentic run holds the single heavy slot | wait, or **Stop run** on the other one (stop is immediate) |
| "Agentic execution degraded" | see the *Checks* list: *agent research (MCP)* failed means the MCP server or agent model was unavailable; *agent evidence IDs* failed means the agent cited IDs no tool returned; the grounded pipeline answer is shown either way | for an MCP failure check `make preflight` and restart `make dev`; otherwise retry |
| Failed: "exceeded its time limit" | the run hit `HARNESS_RUN_DEADLINE_*` | the model is overloaded; retry with the fast budget |
| Failed: "could not propose interpretations" | the agent model is unreachable or returned unusable output | check the Granite alias in preflight |
| Failed: "interrupted" | the API restarted while the run was active | run it again |
| Blank page or "API is not reachable" | the API or tunnel is down | `omen-tunnels ... status`, then restart `make dev` |

Stop is immediate even during a slow model call, and it frees the model slot for the next run.

## 4. Known limitations (say them if asked)

- **Question understanding is model-based.** The planner reads the question and fills in a
  typed plan. An invalid or incomplete plan, or a paper it cannot resolve, falls back to
  full-text research, which is slower but still grounded. A *valid but wrong* plan, for
  example the publication year when the question asked for authors, produces a
  verified fact that does not answer the question. The trace shows the chosen operation,
  and `make eval-planner` measures how often this happens.
- **One API process.** Live progress, active runs and model slots are held in one API
  process. Do not start the API with `--workers` greater than 1. After a restart, runs
  that were active are closed as `interrupted`.
- **Screenshot baselines.** The UI screenshot baselines for `readings`, `failed` and
  `corpus` changed with this release. Refresh them once on the node with
  `cd web && corepack pnpm test:e2e --update-snapshots`.
- **Evaluation sidebar.** For live runs the sidebar's precision and recall fields stay
  empty. Scoring against approved gold happens on the Benchmark page.
