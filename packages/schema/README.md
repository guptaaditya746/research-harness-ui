# schema

The run contract shared by API, UI and evaluation. Pydantic models (`Run`, `Reading`, `ClaimSubgraph`, `Check`, ...). `make fixtures` generates the TypeScript types and fixture runs from them.

## Files

- `models.py` — the contract (imported by labelling; keep the names)
- `generate.py` — fixtures and generated types

## Depends on

nothing (lowest layer).

See `AGENTS.md` for the repository map and conventions.
