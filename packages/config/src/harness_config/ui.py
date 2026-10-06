"""The settings the web UI may change (Settings page), described for its controls.

Each entry names a ``Settings`` field (or one of the run-deadline variables, read dynamically) with a
group, a label and a one-line help text; its kind, options, bounds and default come from the field
itself, so they cannot drift from what :func:`harness_config.settings` accepts.

A change is written into the process environment, so ``settings()`` sees it on its next call: it
applies to runs started afterwards (the API refuses changes while a run is active). The API keeps
the changes in a file (``HARNESS_UI_SETTINGS_FILE``) and applies them again at start-up, over the
``.env`` and the profile. Every run records the values in force (``Run.settings``).
"""

from __future__ import annotations

import json
import os
import typing
from pathlib import Path
from typing import Any, Literal

from pydantic import TypeAdapter, ValidationError

# (group, field or variable, label, help)
CATALOG: tuple[tuple[str, str, str, str], ...] = (
    ("Run input", "run_input", "What runs read",
     ("corpus: each path searches the papers. evidence_doc: the question's evidence document D is the input and "
      "each path picks the triples that answer (docs/RUN_INPUT.md).")),
    ("Run input", "evidence_doc_root", "Evidence documents folder",
     "labelling's evidence_docs folder (<question id>/doc.json); empty = <datasets>/evidence_docs."),
    ("Run input", "evidence_doc_batch", "Triples per pick call", "When D is the input: triples per evidence-review call."),
    ("Run input", "evidence_doc_page", "Triples per D tool result", "Page size of the agents' D tools."),
    ("Models", "model_small", "Pipeline model", "Extraction and answers on Fast and Adaptive (LiteLLM alias)."),
    ("Models", "model_edc", "Open-extraction model", "Empty = the pipeline model."),
    ("Models", "model_agent", "Agent model", "The agent's tool loop."),
    ("Models", "model_high", "High model", "Concept readings and schema inference."),
    ("Models", "model_judge", "Judge / reviewer model", "Evidence review, answer verification, soft-match judge."),
    ("Models", "model_rerank", "Reranker", "A served reranker alias; empty = fused order only."),
    ("Models", "max_output_tokens", "Max output tokens", "Per model reply."),
    ("Time limits", "budget_baseline_fast_seconds", "Pipeline model timeout (fast), s", "Per model call."),
    ("Time limits", "budget_baseline_slow_seconds", "Pipeline model timeout (slow), s", "Per model call."),
    ("Time limits", "budget_agentic_fast_seconds", "Agentic model timeout (fast), s", "Per model call."),
    ("Time limits", "budget_agentic_slow_seconds", "Agentic model timeout (slow), s", "Per model call."),
    ("Time limits", "HARNESS_RUN_DEADLINE_FAST_S", "Run deadline (fast budget), s",
     "Wall-clock limit of a whole run; a path-specific variable in .env still wins."),
    ("Time limits", "HARNESS_RUN_DEADLINE_SLOW_S", "Run deadline (slow budget), s",
     "Wall-clock limit of a whole run; a path-specific variable in .env still wins."),
    ("Time limits", "mcp_agent_timeout_s", "Agent research phase, s", "Time the agent may spend calling tools."),
    ("Evidence per run", "baseline_retrieve_k", "Passages retrieved (Fast)", "Ranked passages the Fast path reads."),
    ("Evidence per run", "adaptive_max_passages", "Max passages (Adaptive)", "Passages the Adaptive path reads at most."),
    ("Evidence per run", "agentic_max_passages", "Max passages (Agentic)", "Passages the Agentic pipeline reads at most."),
    ("Evidence per run", "agentic_max_mcp_seeds", "Agent passage seeds", "Passages the agent hands to the pipeline."),
    ("Evidence per run", "retrieval_mode", "Retrieval", "lexical = BM25; hybrid = BM25 + dense, fused."),
    ("Evidence per run", "claim_reuse", "Reuse earlier claims",
     "Claims accepted in earlier runs become retrieval candidates (leaks between repetitions; off for evaluation)."),
    ("Evidence per run", "subagent_readers", "Paper readers", "Parallel paper readers after the agent's research; 0 = off."),
    ("Evidence per run", "subagent_passages", "Passages per reader", "Passages each paper reader sees."),
    ("Extraction and graph", "edc_batch_passages", "Passages per extraction call",
     "Smaller = more facts per passage, more calls."),
    ("Extraction and graph", "extraction_concurrency", "Parallel extraction calls", "Batches sent at the same time."),
    ("Extraction and graph", "schema_scope", "Schema scope", "focused = what the answer needs; broad = the whole topic."),
    ("Extraction and graph", "graph_linking", "Graph linking",
     "dense = one name per entity, relations between entities, compound statements split."),
    ("Extraction and graph", "unmapped_relations", "Unmapped relations",
     "keep = a fact whose relation fits no schema term gets a new relation; drop = it is lost."),
    ("Extraction and graph", "map_lexical_min", "Mapping: lexical threshold", "0–1."),
    ("Extraction and graph", "map_model_min", "Mapping: model confidence threshold", "0–1."),
    ("Extraction and graph", "schema_new_classes", "Invented classes", "subclass of an existing class, or off."),
    ("Extraction and graph", "dblp_backbone", "DBLP backbone", "Add authors, venue and citations of the answer's papers."),
    ("Agent", "agent_max_tool_calls", "Tool calls", "Tool budget of the agent's loop."),
    ("Agent", "agent_keep_results", "Results kept in full", "Older tool results are shown as digests."),
    ("Agent", "agent_summary_compaction", "Summarise older turns", "Compact the conversation when it grows."),
    ("Agent", "agent_reflections", "Decision corrections", "Times a rejected decision is sent back."),
    ("Agent", "agent_first_tool_choice", "First turn", "required = the first reply must call a tool."),
    ("Agent", "readings_min", "Readings at least", "Concept readings the agent proposes."),
    ("Agent", "readings_max", "Readings at most", "Concept readings the agent proposes."),
    ("Agent", "hide_ungrounded_readings", "Hide ungrounded readings", "Readings without a supporting passage."),
)
# Read outside the Settings class (see Settings.run_deadline_seconds), with their built-in defaults.
DYNAMIC = {"HARNESS_RUN_DEADLINE_FAST_S": 300.0, "HARNESS_RUN_DEADLINE_SLOW_S": 900.0}


def _field(name: str) -> tuple[str, Any, Any, list[Any]]:
    """(environment variable, annotation, default, constraint metadata) of a catalog entry."""
    from . import Settings

    if name in DYNAMIC:
        return name, float, DYNAMIC[name], []
    info = Settings.model_fields[name]
    alias = info.validation_alias
    variable = alias.choices[0] if hasattr(alias, "choices") else str(alias)
    return str(variable), info.annotation, info.default, list(info.metadata)


def _kind(annotation: Any) -> tuple[str, list[str] | None]:
    args = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
    if typing.get_origin(annotation) is Literal:
        return "choice", [str(arg) for arg in typing.get_args(annotation)]
    if args and typing.get_origin(annotation) is not Literal:
        inner = args[0]
        if typing.get_origin(inner) is Literal:
            return "choice", [str(arg) for arg in typing.get_args(inner)]
        annotation = inner
    if annotation is bool:
        return "bool", None
    if annotation is int:
        return "int", None
    if annotation is float:
        return "float", None
    return "text", None


def _bounds(metadata: list[Any]) -> tuple[float | None, float | None]:
    low = next((getattr(item, "ge", None) for item in metadata if getattr(item, "ge", None) is not None), None)
    high = next((getattr(item, "le", None) for item in metadata if getattr(item, "le", None) is not None), None)
    return low, high


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def describe() -> list[dict[str, Any]]:
    """Every UI-editable setting with its group, label, help, kind, options, bounds, default and value."""
    from . import settings

    current = settings()
    rows = []
    for group, name, label, help_text in CATALOG:
        variable, annotation, default, metadata = _field(name)
        kind, options = _kind(annotation)
        low, high = _bounds(metadata)
        if name in DYNAMIC:
            low = 1.0
            value = os.environ.get(variable, "") or _text(default)
        else:
            value = _text(getattr(current, name))
        rows.append({"name": variable, "group": group, "label": label, "help": help_text, "kind": kind,
                     "options": options, "min": low, "max": high, "default": _text(default), "value": value})
    return rows


def values() -> dict[str, str]:
    """{variable: value} of every UI-editable setting (what a run records)."""
    return {row["name"]: row["value"] for row in describe()}


def _validate(name: str, raw: str) -> str:
    entry = next((item for item in CATALOG if _field(item[1])[0] == name), None)
    if entry is None:
        raise ValueError(f"{name} cannot be changed from the UI")
    _, annotation, _, metadata = _field(entry[1])
    text = str(raw).strip()
    if text == "":
        if typing.get_origin(annotation) is not None and type(None) in typing.get_args(annotation):
            return ""
        raise ValueError(f"{name} needs a value")
    target = typing.Annotated[annotation, *metadata] if metadata else annotation
    try:
        value = TypeAdapter(target).validate_python(text)
    except ValidationError as exc:
        raise ValueError(f"{name}: {exc.errors()[0]['msg']}") from None
    if entry[1] in DYNAMIC and value <= 0:
        raise ValueError(f"{name} must be positive")
    return _text(value)


def apply(changes: dict[str, str | None], originals: dict[str, str | None]) -> dict[str, str]:
    """Validate every change, then write them into the environment; ``None`` restores the value from
    before the UI changed it (kept in ``originals``). Returns the overrides now in force."""
    checked = {name: None if value is None else _validate(name, value) for name, value in changes.items()}
    for name, value in checked.items():
        originals.setdefault(name, os.environ.get(name))
        if value is None:
            before = originals.pop(name)
            if before is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = before
        else:
            os.environ[name] = value
    return {name: os.environ.get(name, "") for name in originals}


def load_file(path: Path, originals: dict[str, str | None]) -> dict[str, str]:
    """Apply the overrides saved in ``path`` (missing or unreadable: none). Invalid entries are skipped."""
    try:
        saved = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    applied: dict[str, str] = {}
    for name, value in (saved.items() if isinstance(saved, dict) else []):
        try:
            applied |= apply({str(name): str(value)}, originals)
        except ValueError:
            continue
    return applied


def save_file(path: Path, overrides: dict[str, str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(overrides, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


__all__ = ["CATALOG", "apply", "describe", "load_file", "save_file", "values"]
