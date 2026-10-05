# runs

Run storage and run analysis. Atomic artifact files per run, claim and relation stores, gold candidates, and summaries of saved runs.

## Files

- `store.py` — `ArtifactStore`, `GoldCandidateStore`
- `claims.py`, `relations.py` — SQLite claim store and relation registry
- `analysis.py` — summaries used by `scripts/runs.py`

## Depends on

harness_schema, harness_config.

See `AGENTS.md` in this folder (entry points, imports, test, rules) and the root `AGENTS.md`.
