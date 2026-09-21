.PHONY: fixtures up ingest grobid-native-install ingest-slurm dev test eval web-build

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
	@echo "Run 'uv run uvicorn main:app --app-dir api --reload' and 'cd web && corepack pnpm dev' in separate terminals."

test:
	uv run --python 3.12 pytest -q
	cd web && corepack pnpm build && corepack pnpm test:e2e

eval:
	@echo "Evaluation runner is scheduled for M7."

web-build:
	cd web && corepack pnpm build
