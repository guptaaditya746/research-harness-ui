# Research Harness

M0 is a contract and interface fixture build. The API serves sample runs from
`web/fixtures`; the UI uses them to show the handoff states. Research execution,
corpus ingest, and evaluation arrive in later milestones.

## Setup

Requires Python 3.12, `uv`, Node 20.19+ and `pnpm` via Corepack.

```sh
uv sync --python 3.12 --group dev
corepack pnpm install
make fixtures
```

Copy `.env.example` to `.env` and set the service URLs and credentials for your
environment. `DATABASE_URL` points to PostgreSQL 16; `LANGTRACE_API_HOST` is
the trace ingestion endpoint, not the dashboard URL. Langtrace's SDK sends a
hello span at API startup when either a host or API key is configured.

For the forwarded services shown in this workspace, the relevant ports are
`127.0.0.1:35432` (Postgres) and `127.0.0.1:33004` (tracing). Credentials and
the exact Langtrace ingestion path still come from the service operator.

## Run M0

```sh
uv run --python 3.12 uvicorn main:app --app-dir api --host 127.0.0.1 --port 8000
cd web && corepack pnpm dev
```

Open `http://127.0.0.1:5173`. To inspect a given fixture state directly, use
`?screen=idle`, `running`, `readings`, `agentic_result`, `baseline_result`,
`failed`, `runs`, `corpus`, or `settings`. The evaluation sidebar is part of both
result screens.

M0 is a fixture preview: run buttons open a running log and a fixture result.
They do not call a model. The question text stays in the client unchanged
through scene changes.

## Commands

`make fixtures` regenerates `web/src/generated/run.ts`, JSON Schema and the
four fixture runs from Pydantic models. `make test` runs Python tests, the
TypeScript build, and Playwright screen checks. `make up`, `make ingest`, and
`make eval` are milestone placeholders until their services exist.

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

## M1 corpus work in progress

The M1 ingest command reads PDFs with GROBID, uses Docling for individual PDFs
that have no usable TEI, resolves in-corpus references by DOI then title, writes
structural facts with source IDs, and builds a BM25S index. The first nine corpus
tools are registered in `harness_mcp_corpus.server` for stdio or Streamable HTTP.

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

On this compute node, the Docker socket is inaccessible and `127.0.0.1:8070`
is closed. The single-PDF Docling conversion did not return before it was
stopped, so live ingest of the provided samples has not been verified here.
Fixture-based TEI parsing, citation resolution, BM25 retrieval, search DSL,
and MCP registration are tested with `uv run pytest -q packages/corpus/tests`.

The M1 bare-agent probe exposes only the nine corpus tools to an OpenAI-compatible
chat endpoint. With a completed ingest and a reachable model, run:

```sh
set -a && . ./.env && set +a
HARNESS_MODEL_AGENT=qwen-chat \
uv run --python 3.12 python -m harness_mcp_corpus.probe "Which papers cite p17? Answer from facts and include a fact ID." --root corpus_data
```

The shared `.env` points `LITELLM_BASE_URL` and `HARNESS_MODEL_AGENT_URL` at the
LiteLLM proxy on the same compute node. `qwen-chat` is the default agent, `lite-model` is the small
model, `heavy-model` is the independent judge, and `qwen-embed` is the embedding
model. The loopback URL remains valid when the allocated compute node changes.
On the real 37-paper corpus, `qwen-chat` found no incoming citation edge for `p17`
and identified `p4` as introducing Video Prediction Policy, citing passage
`p4-s20-p1`.
