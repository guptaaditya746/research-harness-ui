# Repository guide for coding agents (Codex, Claude) and people

The Research Harness answers questions about a fixed corpus of scientific papers with three research paths (Fast, Adaptive, Agentic), plus an External baseline (the separate external-harness repo), and shows each answer with its knowledge graph and evidence. Models are served separately by [thesis-llm-setup](https://github.com/guptaaditya746/thesis-llm-setup); gold sets are built in [labelling](https://github.com/guptaaditya746/labelling).

Start with `docs/ARCHITECTURE.md` for how a run flows, `docs/decisions/` for why the code is laid out as it is, and `CONTEXT.md` for the vocabulary (claim subgraph, corpus absence, provisional relation, ...).

Every package has its own `AGENTS.md` (what it owns, entry points, what it may import, how to test it): when a task is inside one package, start there.

## Map: where things are

| Task | Look in |
|---|---|
| Start/list/stop runs, run lifecycle, model slots | `api/runtime.py`, `api/capacity.py`; endpoints in `api/routes_*.py` |
| Fast and Adaptive research pipeline | `packages/baseline/src/harness_baseline/` (`pipeline.py` wires `retrieval.py`, `research_steps.py`, `facts.py`, `outcomes.py`) |
| Question understanding (typed plan) | `packages/baseline/src/harness_baseline/planner.py` |
| Agentic run steps | `packages/agent/src/harness_agent/research.py` |
| Agent tool loop, context window | `packages/agent/src/harness_agent/tool_loop.py`, `decisions.py`, `mcp.py`, `context.py`, `research_state.py` |
| Concept readings before agentic research | `packages/agent/src/harness_agent/concepts.py`, `proposal.py` |
| Model calls (client, retries, cancellation, events) | `packages/extraction/src/harness_extraction/model_client.py`, `call_control.py`, `events.py` |
| Schema inference, extraction, review, answer | `packages/extraction/src/harness_extraction/engine.py` and `stage_*.py` |
| Claim subgraphs (also imported by `labelling`) | `packages/extraction/src/harness_extraction/claims.py` |
| Prompts (every model call) and few-shot examples | `packages/prompts/src/harness_prompts/templates/{v1,v2}/`, `examples/` — see `docs/PROMPTS.md` |
| Settings / environment variables | `packages/config/src/harness_config/__init__.py`, `.env.example` |
| Corpus tools for the agent (MCP) | `packages/mcp_corpus/src/harness_mcp_corpus/server.py` |
| Corpus store, BM25 + dense (hybrid) ranking, ingest | `packages/corpus/src/harness_corpus/` (`store.py`, `dense.py`) |
| Run / API contract (Pydantic, generated TS types) | `packages/schema/src/harness_schema/models.py` → `make fixtures` |
| Ontology seed, OWL/SHACL validation | `packages/semantic/`, `seed/` |
| Evaluation against gold | `packages/evaluation/`, `api/routes_benchmark.py`, `make evaluate`; metric definitions in labelling `docs/EVALUATION.md` |
| Web UI | `web/src/App.tsx`, `web/src/AnswerGraph.tsx` |
| Why a run behaved as it did | `docs/RUN_ANALYSIS.md`, `make runs` |
| Graph size settings, profiles (evidence documents are built in labelling) | `profiles/`, `docs/GRAPH_SETTINGS.md` |
| Known problems and their causes | `docs/KNOWN_ISSUES.md` |

Every package has a short `README.md` saying what it owns and its entry points.

## Layers

Higher layers may import lower ones, never the reverse; `lint-imports` (run by `make check`) enforces this.

```
harness_agent
harness_baseline
harness_extraction | harness_graphstore | harness_mcp_corpus | harness_evaluation | harness_runs
harness_corpus | harness_semantic | harness_observability | harness_ledger | harness_prompts
harness_schema | harness_config
```

`harness_public_gateway` is internet-facing and must not import harness internals; it talks to the API over HTTP.

## Commands

- `make check` — ruff, layer contracts, all Python tests. Run it before every commit.
- `make test` — `make check` plus the web build and Playwright screen tests.
- `make runs ARGS="list -n 20"` — recent runs; `show <id>`, `compare <a> <b>`, `stats`, `--json`.
- `make evaluate` — runs against approved gold: strict / entailed / soft by complexity and path; `ARGS="--judge"` fills the soft-match judge cache.
- `make fewshot ARGS="--dev <ids>"` — few-shot example candidates from gold errors on development questions (`docs/PROMPTS.md`).
- `make embed` — dense passage index for hybrid retrieval, after every ingest (needs `qwen-embed`).
- `make graph-shape` — answer-graph shape of recent runs (read-only; Phase 4 diagnostics).
- `make run-questions` — evaluation questions through the running API (`--questions` for nlp32–61); Agentic runs every proposed reading.
- `make fixtures` — regenerate `web/src/generated/run.ts`, JSON Schema and fixture runs after changing `harness_schema`.
- `make dev` / `make demo` — MCP server, API and UI (needs the models from thesis-llm-setup).
- `make preflight` — checks config, corpus, models, model status, MCP and API on the live node.
- One package's tests: `uv run --python 3.12 pytest -q packages/<name>`.

## Conventions

- Settings: read with `from harness_config import settings; settings().<field>`. Never call `os.getenv` for harness settings. A new variable needs a field in `Settings` and a line in `.env.example` (a test checks both).
- Prompts: never write prompt text in Python. Render a template: `complete(model, render("area/name", payload))`. Change v2 templates; keep v1 frozen (golden-tested). See `docs/PROMPTS.md`.
- Model calls go through LiteLLM aliases (`heavy-model`, `lite-model`, `qwen-embed`), never raw vLLM ports.
- Evidence ids: only ids a tool or the corpus actually returned may reach claims or answers.
- Triples and graphs follow RDF / OWL 2 (IRIs, `rdfs:domain`/`rdfs:range`, `rdfs:subPropertyOf`); invented terms live in the `hrn:` namespace.
- Big modules are split by responsibility; public names are re-exported from each package `__init__.py`. Keep imports like `from harness_extraction import X` working.
- Tests use stub models (`complete_json(system, user)`); no test calls a real model or GPU.

## Do not

- Commit `.env`, tokens or keys, or paste them into logs.
- Add anything specific to the evaluation questions or their papers: no code paths, regexes or keyword lists
  keyed on a question, paper, benchmark or metric, and no prompt text or examples that copy or paraphrase an
  evaluation question (`datasets/question_library_2025*.json`). Prompt examples use made-up entities
  (p99, MethodX, BenchY); a new capability must be general and tested on wording outside the evaluation set.
- Edit `web/src/generated/` or fixture runs by hand (use `make fixtures`).
- Add a setting that only restores old behaviour for comparison (`docs/decisions/0007`): compare against a tag instead.
- Rename `harness_extraction.claims.build_claims` or `harness_schema.models` — `labelling` imports them.
- Call raw vLLM ports for model requests, or add fallbacks between model roles.
- Edit thesis-llm-setup or labelling from this repository.

## Which document to read

- How a run flows, model roles, context flow: `docs/ARCHITECTURE.md`.
- Inspecting runs: `docs/RUN_ANALYSIS.md`. Known problems: `docs/KNOWN_ISSUES.md`.
- Prompts: `docs/PROMPTS.md`. Live session: `docs/DEMO.md`. Decisions: `docs/adr/`.
