"""Versioned prompt templates for every model call in the harness.

Each model call has one named prompt, e.g. ``extraction/schema_inference``.
Its text lives in ``templates/<set>/<name>.system.j2`` (and, for calls whose
user message is prose rather than data, ``<name>.user.j2``). Two sets exist:

- ``v1``: the prompts exactly as the harness sent them before templating
  (golden-tested byte for byte), kept so runs can be compared.
- ``v2``: rewritten to current prompt-writing practice for non-thinking
  instruct models (role and task first, rules with reasons, explicit output
  schema, what to do when evidence is missing, documents before the question).

``HARNESS_PROMPT_SET`` selects the set (default ``v2``). Every rendered prompt
carries ``ref`` = ``<name>@<set>:<hash>``; model events and traces record it,
so a result can always be traced to the exact prompt text that produced it.

Few-shot examples (Phase 3f): ``examples/<set>/<name>.json`` (a list of
``{"input": ..., "output": ..., "why": ...}``) is passed to the template as
``examples``; v2 templates of the extraction, answer and reader prompts show
them when present. The examples are part of the prompt's hash. Choose them from
failures on a development part of the gold set, never from the questions you
report on (see docs/PROMPTS.md).

Data payloads stay JSON (they are machine-readable and the IDs must be
copied exactly). In v2 the long parts of a payload (passages, candidates,
triples, claims) come first and the question comes last, which is the order
long-context guidance recommends.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import Any

from harness_config import settings
from jinja2 import Environment, StrictUndefined

SETS = ("v1", "v2")

# Payload keys moved to the front (long material) and to the end (the ask) in v2.
_DOCUMENT_KEYS = (
    "passages", "candidates", "triples", "claims", "cited_claims", "cited_evidence",
    "unresolved", "relations", "evidence_not_found_in_session",
)
_LAST_KEYS = ("research_interpretation", "research_notes", "agent_decision", "draft_answer", "question")


@dataclass(frozen=True)
class Prompt:
    """One rendered prompt: the messages to send and where they came from."""

    name: str
    prompt_set: str
    version: str
    system: str
    user: str

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.prompt_set}:{self.version}"


@cache
def _environment() -> Environment:
    return Environment(
        undefined=StrictUndefined,
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=False,
    )


@cache
def _examples(prompt_set: str, name: str) -> tuple[str, list[dict[str, Any]]]:
    """The raw JSON (for the hash) and the parsed examples for one prompt; empty when absent."""
    path = resources.files("harness_prompts").joinpath("examples", prompt_set, f"{name}.json")
    try:
        raw = path.read_text(encoding="utf-8")
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
        return "", []
    examples = json.loads(raw)
    if not isinstance(examples, list) or not all(
            isinstance(item, dict) and {"input", "output"} <= set(item) for item in examples):
        raise ValueError(f"examples for {name!r} in set {prompt_set!r} must be a list of objects "
                         "with input and output (why is optional)")
    return raw, examples


@cache
def _source(prompt_set: str, name: str, part: str) -> str | None:
    path = resources.files("harness_prompts").joinpath("templates", prompt_set, f"{name}.{part}.j2")
    try:
        return path.read_text(encoding="utf-8")
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
        return None


def _resolve_set(prompt_set: str | None) -> str:
    chosen = (prompt_set or settings().prompt_set or "v2").strip()
    if chosen not in SETS:
        raise ValueError(f"unknown prompt set {chosen!r}; choose one of {', '.join(SETS)}")
    return chosen


def _render_text(source: str, variables: dict[str, Any]) -> str:
    return _environment().from_string(source).render(**variables)


def _ordered(payload: dict[str, Any]) -> dict[str, Any]:
    first = {key: payload[key] for key in _DOCUMENT_KEYS if key in payload}
    last = {key: payload[key] for key in _LAST_KEYS if key in payload}
    middle = {key: value for key, value in payload.items() if key not in first and key not in last}
    return {**first, **middle, **last}


def render(
    name: str,
    payload: dict[str, Any] | None = None,
    *,
    ensure_ascii: bool = True,
    prompt_set: str | None = None,
    **variables: Any,
) -> Prompt:
    """Render the system message (and user message) for one model call.

    ``payload`` is the call's data. Without a ``user`` template it is sent as
    JSON: in the original key order in v1, documents first and question last
    in v2. ``variables`` fill placeholders in the templates.
    """
    chosen = _resolve_set(prompt_set)
    system_source = _source(chosen, name, "system")
    if system_source is None:
        raise KeyError(f"no prompt {name!r} in set {chosen!r}")
    user_source = _source(chosen, name, "user")
    examples_raw, examples = _examples(chosen, name)
    variables = {"examples": examples, **variables}
    system = _render_text(system_source, variables)
    if user_source is not None:
        user = _render_text(user_source, {"payload": payload or {}, **variables})
    elif payload is not None:
        data = payload if chosen == "v1" else _ordered(payload)
        user = json.dumps(data, ensure_ascii=ensure_ascii)
    else:
        user = ""
    digest = hashlib.sha256(
        (system_source + "\x00" + (user_source or "") + ("\x00" + examples_raw if examples_raw else "")).encode()
    ).hexdigest()[:8]
    return Prompt(name=name, prompt_set=chosen, version=digest, system=system, user=user)


def text(name: str, *, prompt_set: str | None = None, **variables: Any) -> str:
    """Render a single-message template (for example a message inside a conversation)."""
    return render(name, prompt_set=prompt_set, **variables).system


def names(prompt_set: str = "v1") -> list[str]:
    """Every prompt name defined in a set."""
    root = resources.files("harness_prompts").joinpath("templates", prompt_set)
    found: list[str] = []

    def walk(node: Any, prefix: str) -> None:
        for child in node.iterdir():
            if child.is_dir():
                walk(child, f"{prefix}{child.name}/")
            elif child.name.endswith(".system.j2"):
                found.append(prefix + child.name.removesuffix(".system.j2"))

    walk(root, "")
    return sorted(found)


__all__ = ["SETS", "Prompt", "names", "render", "text"]
