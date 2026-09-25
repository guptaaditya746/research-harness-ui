# prompts

Versioned prompt templates for every model call. `render(name, payload)` returns a `Prompt` (system, user, `ref`). Sets v1 (legacy, golden-tested) and v2 (current). See `docs/PROMPTS.md`.

## Files

- `__init__.py` — `render`, `text`, `names`, `Prompt`
- `templates/v1`, `templates/v2` — the templates
- `examples/v2/<area>/<name>.json` — optional few-shot examples, shown by five v2 templates and part of the hash (`examples/README.md`)
- `tests/golden/` — the exact pre-template prompts (do not edit)

## Depends on

harness_config.

See `AGENTS.md` for the repository map and conventions.
