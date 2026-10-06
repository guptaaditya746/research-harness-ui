# harness_baseline

The Pipeline paths (Fast, Adaptive) and the steps the Agentic path reuses: question understanding, fact operations, retrieval, schema inference, extraction, the answer graph and the run's artifacts.

## Entry points

- `pipeline.py` — `Baseline20.run`
- `retrieval.py`, `research_steps.py`, `facts.py` (one method per fact operation, `FACT_OPERATIONS`), `outcomes.py`
- `run_graph.py` — the run's RDF provenance graph; `answer_graph.py` — the UI graph
- `planner.py` — query plans; `external.py` — the External harness adapter
- `evidence_doc.py`, `given_evidence.py` — the evidence document D as a run's input (docs/RUN_INPUT.md)

## Imports

Layer 2 of 5 (`[tool.importlinter]` in the root `pyproject.toml`): may import harness_extraction, harness_graphstore, harness_evaluation, harness_runs, harness_mcp_corpus and lower layers (not harness_agent).

## Test

```sh
uv run --python 3.12 pytest -q packages/baseline
```

## Rules

- A new fact operation needs an entry in `planner.OPERATIONS` and in `facts.FACT_OPERATIONS`.
- Artifact names are a contract with `harness_schema.models.required_artifacts` and the UI.

Repository-wide rules and the map: the root `AGENTS.md`.
