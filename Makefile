.PHONY: fixtures up ingest grobid-native-install ingest-slurm dev test eval web-build preflight eval-planner

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

test:
	uv run --python 3.12 pytest -q
	cd web && corepack pnpm build && corepack pnpm test:e2e

eval:
	uv run --python 3.12 harness-eval run --artifacts $${ARTIFACT_ROOT:-artifacts} --dataset $${DATASET:-datasets/dev.jsonl} $(if $(PATH_FILTER),--path $(PATH_FILTER),)

web-build:
	cd web && corepack pnpm build

preflight:
	uv run --python 3.12 python scripts/preflight.py

eval-planner:
	uv run --python 3.12 python scripts/eval_planner.py
