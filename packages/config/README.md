# config

Typed settings for every environment variable the harness reads. `settings()` returns a fresh `Settings` (pydantic-settings) from the environment; the nearest `.env` is loaded on import without overriding set variables.

## Files

- `__init__.py` — `Settings`, `settings()`, `run_deadline_seconds`, `model_options`
- `tests/test_config.py` — keeps `.env.example` and `Settings` in sync

## Depends on

nothing (lowest layer).

See `AGENTS.md` for the repository map and conventions.
