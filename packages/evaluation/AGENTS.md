# harness_evaluation

Scoring runs against labelling's gold: triple precision and recall at three levels, vital recall, judged precision, the benchmark by complexity × path, graph-shape diagnostics.

## Entry points

- `matching.py` — `score_levels` (pairing, judge, per-level scores)
- `relations.py` — entailment rules; `judge.py` — the soft-match judge
- `benchmark.py`, `stats.py`, `graph_shape.py`
- `explain.py` — why a gold row scores zero or nothing (`make evaluate ARGS=--explain`)

## Imports

Layer 3 of 5 (`[tool.importlinter]` in the root `pyproject.toml`): may import the layers below (schema, config, prompts, semantic, ...), not its siblings.

## Test

```sh
uv run --python 3.12 pytest -q packages/evaluation
```

## Rules

- Changing scoring changes reported numbers: show identical `make evaluate` output for refactors, and record real changes in labelling's docs/EVALUATION.md.

Repository-wide rules and the map: the root `AGENTS.md`.
