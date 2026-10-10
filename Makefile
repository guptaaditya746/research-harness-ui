.PHONY: pipeline-map public ontology cskg-subset benchmark-runs run-questions graph-shape fewshot embed fixtures up ingest grobid-native-install ingest-slurm dev demo check test eval web-build preflight eval-planner runs evaluate

fixtures:
	uv run --python 3.12 python -m harness_schema.generate

up:
	docker compose -f docker/compose.yml up -d grobid

ingest:
	uv run --python 3.12 python -m harness_corpus $(CORPUS) --root $${CORPUS_ROOT:-corpus_data}

grobid-native-install:
	./scripts/bootstrap_grobid_native.sh

ingest-slurm:
	@test -n "$(CORPUS)" || (echo "usage: make ingest-slurm CORPUS=/path/to/pdfs" >&2; exit 2)
	mkdir -p slurm/logs
	sbatch --export=ALL,CORPUS="$(CORPUS)",PROJECT_DIR="$(CURDIR)" slurm/grobid_ingest.sbatch

dev:
	./scripts/dev.sh

demo:
	HARNESS_DEV_RELOAD=0 ./scripts/dev.sh

# The thesis pipeline map (docs/presentation/README.md): a static page, served for a browser elsewhere.
# make pipeline-map · omen-tunnels pipeline-map route uses compute port 8099.
PORT ?= 8099
pipeline-map:
	@echo "Open http://localhost:$(PORT)/thesis_pipeline.html  (Ctrl+C to stop)"
	@echo "Laptop via omen-tunnels (default port 8099): http://127.0.0.1:28099/thesis_pipeline.html"
	python3 -m http.server $(PORT) --bind 127.0.0.1 --directory docs/presentation

# Fast Python-only check for every change: lint, layer contracts, tests.
check:
	uv run --python 3.12 ruff check .
	uv run --python 3.12 lint-imports
	uv run --python 3.12 pytest -q

test: check
	cd web && corepack pnpm build && corepack pnpm test:e2e

eval:
	uv run --python 3.12 harness-eval run --artifacts $${ARTIFACT_ROOT:-artifacts} --dataset $${DATASET:-datasets/dev.jsonl} $(if $(PATH_FILTER),--path $(PATH_FILTER),)

web-build:
	cd web && corepack pnpm build

preflight:
	uv run --python 3.12 python scripts/preflight.py

eval-planner:
	uv run --python 3.12 python scripts/eval_planner.py

# The full harness (all modes and pages) on the internet behind one password, via a free Cloudflare quick tunnel
public:
	./scripts/public-full.sh

# The seed ontology as Turtle, for the external harness (its knowledge_base.ontology)
ontology:
	uv run --python 3.12 python scripts/export_ontology.py $(ARGS)

# Rebuild seed/cskg_ontology_2.0_subset.ttl from seed/cskg_ontology_2.0.ttl (ARGS="--check" or "--source file-or-url")
cskg-subset:
	uv run --python 3.12 python scripts/cskg_subset.py $(ARGS)

# Summaries of recent runs (see docs/RUN_ANALYSIS.md): make runs ARGS="list -n 20"
runs:
	uv run --python 3.12 python scripts/runs.py $(if $(ARGS),$(ARGS),list)

# Dense passage index for hybrid retrieval (needs the embedding model from thesis-llm-setup).
embed:
	uv run --python 3.12 python -m harness_corpus --embed --root $${CORPUS_ROOT:-corpus_data}

# Gold-based scores: strict / entailed / soft by complexity and path (labelling/docs/EVALUATION.md).
# make evaluate ARGS="--judge" also asks the soft-match judge and caches its answers.
evaluate:
	uv run --python 3.12 python scripts/evaluate.py $(ARGS)

# Scoring runs for the approved gold: every question and path, 3 repetitions, evaluation mode
# make benchmark-runs ARGS="--ids nlp11 --paths agentic,external" (see scripts/run_benchmark.py)
benchmark-runs:
	uv run --python 3.12 python scripts/run_benchmark.py $(ARGS)

# The 30 evaluation questions through the running API; Agentic runs every proposed reading.
# make run-questions ARGS="--ids nlp07,nlp27" (see scripts/run_questions.py)
run-questions:
	uv run --python 3.12 python scripts/run_questions.py $(ARGS)

# Shape of recent answer graphs: DBLP/invented mix, connectivity, dead ends (read-only, Phase 4).
# make graph-shape ARGS="-n 100 --path agentic"
graph-shape:
	uv run --python 3.12 python scripts/graph_shape.py $(ARGS)

# Few-shot example candidates from gold errors on the development questions (docs/PROMPTS.md):
# make fewshot ARGS="--dev nlp01,nlp04"
fewshot:
	uv run --python 3.12 python scripts/fewshot_candidates.py $(ARGS)
