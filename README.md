# Research Harness

The harness includes the fixture contract, live baseline and enhanced retrieval,
agentic research, corpus tools, artifact evaluation, and the React UI. Runs use
the same corpus and provenance checks across all paths.

For a live session, follow [docs/DEMO.md](docs/DEMO.md): `make dev` starts the MCP corpus
server, API and UI; `make preflight` checks every dependency; `make eval-planner` measures
question understanding on the live model.

## Setup

Requires Python 3.12, `uv`, Node 20.19+ and `pnpm` via Corepack.

```sh
uv sync --python 3.12 --group dev
corepack pnpm install
make fixtures
```

Copy `.env.example` to `.env` and set the service URLs and credentials for your
environment. `DATABASE_URL` points to PostgreSQL 16; `LANGFUSE_BASE_URL` is
the Langfuse project URL. Langfuse sends a hello span at API startup when both
project keys are configured.

For the forwarded services shown in this workspace, the relevant ports are
`127.0.0.1:35432` (Postgres). Langfuse credentials and the project URL belong
in `.env` and should never be committed.

## Run M0

```sh
uv run --python 3.12 uvicorn main:app --app-dir api --host 127.0.0.1 --port 8000
cd web && corepack pnpm dev
```

Open `http://127.0.0.1:5173`. To inspect a given fixture state directly, use
`?screen=idle`, `running`, `readings`, `agentic_result`, `baseline_result`,
`failed`, `runs`, `corpus`, or `settings`. The evaluation sidebar is part of both
result screens.

### Browser access from a Slurm node

The repository includes an `omen-tunnels` version with a reverse `harness-ui`
route. Install the same script at `~/.local/bin/omen-tunnels` on the laptop and
the compute node. Start all routes on each machine:

```sh
# Laptop, once after login or reboot
omen-tunnels laptop start all

# Inside the Slurm allocation
omen-tunnels compute start all
make dev
```

Open `http://127.0.0.1:25173` on the laptop. The route carries browser traffic
to Vite on compute port 5173; Vite proxies API requests to compute port 8000.
No separate `ssh -L` command is needed. Stop and restart both tunnel commands
after changing their selected service list.

M0 is a fixture preview: run buttons open a running log and a fixture result.
They do not call a model. The question text stays in the client unchanged
through scene changes.

## Commands

`make fixtures` regenerates `web/src/generated/run.ts`, JSON Schema and the
fixture runs from Pydantic models. `make test` runs Python tests, the TypeScript
build, and Playwright screen checks. `make up` starts GROBID, `make ingest`
builds corpus artifacts, and `make eval DATASET=...` evaluates completed run
artifacts with the pure evaluator.

The live API supports `POST /runs`, SSE at `/runs/{id}/events`, reading
selection at `POST /runs/{id}/reading`, cooperative cancellation at
`POST /runs/{id}/stop`, artifact evaluation at `POST /runs/{id}/evaluate`, and
`GET|PUT /settings`.

Each live run creates a Langfuse parent observation (`research.run`) and child
stage observations for retrieval, schema inference, extraction, and answer
assembly. The local `trace.json` artifact and SSE stream remain the UI-facing
trace; exporter failures do not fail the research run.

## Artifact storage and DB

`ArtifactStore` writes each run's required artifact files under `ARTIFACT_ROOT`,
including explicit empty files on terminal failure, then writes a SHA-256
manifest. The initial PostgreSQL schema is in `packages/runs/alembic` and can
be installed with:

```sh
uv run --python 3.12 alembic -c packages/runs/alembic.ini upgrade head
```

The M0 fixture API reads generated JSON files; connecting live run rows to
Postgres is part of the next runtime milestone. No model or sample corpus is
needed to view the fixture UI.

## GROBID on Slurm without Docker

The Slurm job runs GROBID and ingestion together on one CPU node, so GROBID only
listens on that job's loopback interface. On the first run it downloads a
user-space Temurin JDK 21, builds the pinned GROBID 0.9.0 release, and stores the
installation under `.services/`. Later jobs reuse it.

```sh
make ingest-slurm \
  CORPUS=/mnt/ceph/storage/data-tmp/current/xepi3167/thesis/final-final-frontend/samples
```

Watch the job with `squeue --me`. Slurm output is written under `slurm/logs/`,
GROBID's service log is `slurm/logs/grobid-service-<job-id>.log`, and the corpus
artifacts are written to `corpus_data/` by default. Override the destination by
exporting `CORPUS_ROOT` before submitting. The native bootstrap can also be run
inside any allocation with `make grobid-native-install`.

This route needs outbound HTTPS during the first allocation so Gradle, GROBID,
and JDK dependencies can be downloaded. It uses the CRF models and CPUs, matching
the previously pinned Docker setup.

## M1 corpus

The immutable ontology seed is `seed/dblp_schema_2024-06-14.ttl`, the official
DBLP RDF schema. Enhanced run artifacts record its version and SHA-256. Verified
paper individuals use their DBLP record IRIs in graph artifacts; internal `pN`
keys remain only in corpus storage and provenance.

The M1 ingest command reads PDFs with GROBID, uses Docling for individual PDFs
that have no usable TEI, resolves in-corpus references by DOI then title, writes
structural facts with source IDs, and builds a BM25S index. The corpus tools are
registered in `harness_mcp_corpus.server` for stdio or Streamable HTTP, including
retrieval, expansion, keyphrases, results, tables, sections, citation contexts,
ranking, entities, prior triples, and corpus maps.

```sh
make up
make ingest CORPUS=/path/to/pdfs
uv run --python 3.12 python -m harness_mcp_corpus.server
```

`make up` starts the pinned GROBID CRF container on port 8070. The corpus tool
server reads `CORPUS_ROOT` (default `corpus_data`); set
`MCP_CORPUS_TRANSPORT=stdio` to run it as a local subprocess. Ingest writes
`ingest_report.json`, including papers whose DBLP record could not be verified.
Each paper has a stable identifier. Verified DBLP matches use `dblp:` and
`identifier_source: dblp`; unmatched papers use `local:paper/<sha256-prefix>`
and `dblp_status: unresolved`. The local ID is derived from PDF content and is
stable across ingest runs. A missing DBLP IRI is recorded as `null`, never
guessed. This is the user-approved M1 identifier rule; the original plan's
all-DBLP requirement has been superseded.

The current sample ingest contains 37 DBLP-verified papers, 2,683 passages,
2,384 references, structural facts, outlines, and a BM25S index. Fixture-based
TEI parsing, citation resolution, BM25 retrieval, search DSL, and MCP
registration are tested with `uv run pytest -q packages/corpus/tests`.

The M1 bare-agent probe exposes the corpus tools to an OpenAI-compatible chat
endpoint. With a completed ingest and a reachable model, run:

```sh
set -a && . ./.env && set +a
HARNESS_MODEL_AGENT=qwen-chat \
uv run --python 3.12 python -m harness_mcp_corpus.probe "Which papers cite p17? Answer from facts and include a fact ID." --root corpus_data
```

The shared `.env` points `LITELLM_BASE_URL` and `HARNESS_MODEL_AGENT_URL` at the
LiteLLM proxy on the same compute node. `qwen-chat` is the default extraction model;
`HARNESS_MODEL_HIGH=heavy-model` selects the higher-quality supervisor and
evidence reviewer for agentic runs. The small model extracts candidate triples;
the reviewer selects question-relevant IDs from those candidates and verifies
the drafted answer against their evidence. Verified metadata questions use
corpus facts directly. `HARNESS_MAX_CONCURRENT_RUNS` and
`HARNESS_MAX_CONCURRENT_HEAVY` queue model calls when evaluation starts many
repetitions at once.
Set `HARNESS_MODEL_SMALL` to another alias only when its vLLM endpoint is running. `heavy-model`
is also available as the independent judge, and `qwen-embed` is the embedding
model. The loopback URL remains valid when the allocated compute node changes.
On the real 37-paper corpus, `qwen-chat` found no incoming citation edge for
`p17` and identified `p4` as introducing Video Prediction Policy, citing
`p4-abstract`. The recorded calls and tool results are under `docs/evidence/`.

The web API's agentic path connects to the managed Streamable HTTP server at
`MCP_CORPUS_URL`. Start the MCP service separately before launching an agentic
run. The supervisor discovers the registered corpus tools and lets retrieval,
structure, and verification roles call them over one session. Complete tool-call
records are persisted as `tool_calls.jsonl`; compact events are streamed and
written to `trace.json`; Langfuse receives one child observation per tool call.
The API does not impose a corpus-wide result or triple cap. MCP's page size is
50, and the supervisor can request additional pages when a tool reports more
results.

## M2 semantic core

`harness_ledger` owns monotonic coverage obligations and closes structural
obligations from ingest facts. `harness_semantic` supplies the hand-written
subschema, SHACL property shapes, and OWL RL closure. `harness_graphstore`
checks SPARQL terms against that schema, derives coverage plans from query
shape and the fact domain, and stores extraction events in provenance-bearing
named graphs. The M2 real-corpus closure evidence is
`docs/evidence/m2-structural-ledger.json`.

## Live runs and evaluation

The baseline and enhanced paths use lexical/structural retrieval and EDC
extraction. Agentic runs add planner lenses and shared verification artifacts.
Completed runs are stored under `ARTIFACT_ROOT` with explicit artifact sets and
SHA-256 manifests, including empty files for failed terminal runs.

To score stored runs offline, create a JSONL dataset with `question` and
`gold_triples`, then run:

```sh
uv run harness-eval run --artifacts artifacts --dataset datasets/dev.jsonl
```

The evaluator computes metrics from stored triples and never trusts model
reported metrics. It writes a timestamped JSON report and prints a compact
comparison table.

## Question understanding

Every run starts with `Understanding question`. The planner (`harness_baseline.planner`)
asks the extraction model for a typed `QueryPlan` (operation, paper mentions, year,
extreme, rank or minimum paper count, evidence focus, benchmark), validates it, and
resolves paper mentions against corpus titles. Metadata, citation and aggregation
operations are answered from verified facts. Everything else goes through full-text
research, scoped by the resolved papers and the plan's evidence focus. Invalid plans,
unresolved papers and uncovered facts fall back to research visibly in the trace. The
question's wording never selects a route.

On the agentic path the agent first proposes 3–4 concept readings, each grounded with
corpus search hits. The chosen concept travels with the unchanged question as structured
input (`Run.concept`); picking is idempotent, and each child run records `parentRunId`.
Stop and the per-run deadline cancel runs cooperatively, including model calls that are
already in flight.
