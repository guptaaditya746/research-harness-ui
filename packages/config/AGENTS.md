# harness_config

Settings: every environment variable, its default and profiles.

## Entry points

- `__init__.py` — `settings()`, `Settings`, `load_env_file`, `profile_file`
- `ui.py` — the settings the Settings page may change (`CATALOG`), validated and applied to the environment

## Imports

Layer 5 of 5 (`[tool.importlinter]` in the root `pyproject.toml`): may import nothing from the harness.

## Test

```sh
uv run --python 3.12 pytest -q packages/config
```

## Rules

- A new setting goes here and into `.env.example` (the config test checks both); add it to `ui.CATALOG` when people should change it from the UI.

Repository-wide rules and the map: the root `AGENTS.md`.
