"""Agent-proposed concept readings shown before every agentic research run."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

MIN_CONCEPTS = 3
MAX_CONCEPTS = 4

SYSTEM_PROMPT = (
    "You are the planning agent of a research harness over a fixed corpus of scientific papers. "
    "Before any research runs, propose between {min} and {max} distinct concept readings of the "
    "user's question. Each reading is one way to interpret what the question is asking for: which "
    "concept, criterion, or relation should be researched. Readings must differ in substance, not "
    "wording. Use only the allowed ontology classes. Return strict JSON: "
    '{{"concepts": [{{"label": str (<= 8 words), "gloss": str (one sentence saying what will be '
    'researched and how it is measured), "classes": [allowed class names], '
    '"search_terms": [2-5 short corpus search phrases]}}]}}'
)


class ConceptProposalError(RuntimeError):
    """The agent could not produce a valid set of concept readings."""


@dataclass
class Concept:
    label: str
    gloss: str
    classes: list[str]
    search_terms: list[str]
    evidence_hits: int = 0
    sample_sources: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "gloss": self.gloss,
            "classes": self.classes,
            "search_terms": self.search_terms,
            "evidence_hits": self.evidence_hits,
            "sample_sources": self.sample_sources,
        }


def _clean(text: object, limit: int) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()[:limit]


def _parse(payload: dict[str, Any], allowed_classes: set[str]) -> list[Concept]:
    raw = payload.get("concepts") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        return []
    concepts: list[Concept] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        label = _clean(item.get("label"), 80)
        gloss = _clean(item.get("gloss"), 300)
        key = label.casefold()
        if not label or not gloss or key in seen:
            continue
        seen.add(key)
        classes = [
            str(name) for name in item.get("classes") or []
            if isinstance(name, str) and name in allowed_classes
        ]
        terms = [
            _clean(term, 60) for term in item.get("search_terms") or []
            if isinstance(term, str) and _clean(term, 60)
        ][:5]
        concepts.append(Concept(label, gloss, classes, terms))
    return concepts


def _ground(concept: Concept, corpus: object | None) -> None:
    """Count corpus passages matching any of the concept's search phrases."""
    search = getattr(corpus, "search", None)
    if search is None:
        return
    sources: list[str] = []
    total = 0
    for term in concept.search_terms:
        words = re.findall(r"\w+", term)
        if not words:
            continue
        query = '"' + " ".join(words) + '"' if len(words) > 1 else words[0]
        try:
            result = search(query, limit=3)
        except Exception:  # noqa: BLE001, S112 - grounding is advisory, never fatal
            continue
        total += int(result.get("total", 0))
        sources.extend(
            str(row.get("passage_id") or row.get("source_id"))
            for row in result.get("items", [])
            if row.get("passage_id") or row.get("source_id")
        )
    concept.evidence_hits = total
    concept.sample_sources = list(dict.fromkeys(sources))[:5]


def propose_concepts(
    model: object,
    question: str,
    allowed_classes: set[str],
    corpus: object | None = None,
    attempts: int = 2,
) -> list[Concept]:
    """Ask the agent model for 3-4 concept readings and ground them in the corpus.

    Concepts with corpus evidence are ranked first. Ungrounded concepts are kept
    only when needed to reach the minimum, so the user always sees 3-4 options.
    """
    complete = getattr(model, "complete_json", None)
    if complete is None:
        raise ConceptProposalError("agent model cannot produce structured output")
    minimum = int(os.getenv("READINGS_MIN", MIN_CONCEPTS))
    maximum = int(os.getenv("READINGS_MAX", MAX_CONCEPTS))
    minimum, maximum = max(MIN_CONCEPTS, minimum), min(MAX_CONCEPTS, max(maximum, MIN_CONCEPTS))
    system = SYSTEM_PROMPT.format(min=minimum, max=maximum)
    last_error = "no concepts returned"
    request: dict[str, Any] = {"question": question, "allowed_classes": sorted(allowed_classes)}
    for attempt in range(attempts):
        if attempt:
            # Tell the model what was wrong instead of repeating an identical
            # deterministic request.
            request["previous_attempt_problem"] = last_error
        user = json.dumps(request)
        try:
            concepts = _parse(complete(system, user), allowed_classes)
        except Exception as exc:  # noqa: BLE001 - provider errors are reported below
            last_error = f"{type(exc).__name__}: {exc}"
            continue
        if len(concepts) < minimum:
            last_error = f"agent proposed {len(concepts)} concepts; {minimum} required"
            continue
        for concept in concepts:
            _ground(concept, corpus)
        concepts.sort(key=lambda item: item.evidence_hits == 0)
        return concepts[:maximum]
    raise ConceptProposalError(last_error)


__all__ = ["Concept", "ConceptProposalError", "propose_concepts"]
