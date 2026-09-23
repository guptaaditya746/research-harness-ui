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
| `HARNESS_MCP_AGENT_TIMEOUT_S` | `240` | the agent's tool loop (never below two agent turns) |
| `HARNESS_RUN_DEADLINE_FAST_S` / `_SLOW_S` | `300` / `900` | hard limit per run; the run ends as `deadline_exceeded` |
| `HARNESS_AGENT_CONTEXT_TOKENS` | the Granite `max_model_len` | the agent's conversation is fitted to this window; if unset it is asked from LiteLLM or learned from the first length error |

Start the laptop tunnel and the stack:

```sh
omen-tunnels laptop start all        # on the laptop
omen-tunnels compute start all       # on the compute node
make dev                             # MCP corpus server + API + UI
```

In a second shell on the node, run both checks. Fix every `FAIL` before you present:

```sh
make preflight       # config, corpus, each model alias, Granite tool calling, MCP, API, storage
make eval-planner    # question understanding on 29 phrasings; should be >= 85 %
```

If `eval-planner` misclassifies a question type, write it down: questions of that type will
fall back to full-text research. They still get an answer, only a slower one.

Open `http://127.0.0.1:25173` on the laptop.

## 2. What to show

No typing is needed. Open **Question library** under the question box. Hover a question to
run it with **Fast**, **Adaptive**, **Agentic** or **Compare**; click its text to put it in the box and
edit it first. The library lists the question matrix (`HARNESS_QUESTION_SETS`, default
`question_matrix_20.json`), gold candidates and your recent live runs. Filter by any word.

The corpus has 37 DBLP-verified papers. These questions come from
`datasets/question_matrix_20.json`.

| Show | Path | Question | Expected |
|---|---|---|---|
| Question understanding | Fast | *Who authored FOCUS: object-centric world models for robotic manipulation?* | answered from verified author facts; trace starts with `Understanding question · paper_authors` |
| Same meaning, different wording | Fast | *List the people who wrote FOCUS.* | same answer and same trace, because the plan routes the question, not the wording |
| Honest absence | Fast | *Which papers cite the Video Prediction Policy paper?* | "No in-corpus match found": `corpus_absent` on the citation index, not a guess |
| Full-text research | Adaptive | *What improvement did Video Prediction Policy report on CALVIN ABC→D?* | 41.5 % result, cited to a passage, with the claim subgraph |
| Agent concepts | Agentic | *Who is the best author in the corpus?* | the agent proposes 3–4 interpretations with corpus evidence counts; pick one |
| Agentic research | Agentic | *What limitation of learning on physical robots does the DayDreamer paper report?* | pick a concept → MCP tool calls in the trace → verified answer |
| Side by side | Compare all | *Which paper introduced Video Prediction Policy?* | fast and adaptive finish; then you choose the agentic concept |

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
| "Queued for model capacity" in the trace | another agentic run holds the single heavy slot | wait, or **Stop run** on the other one (stop is immediate) |
| "Agentic execution degraded" | MCP server or agent model unavailable; you see the grounded baseline answer | check `make preflight`; restart `make dev` |
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
