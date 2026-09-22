"""Pure artifact evaluation for research runs."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from urllib.parse import unquote, urlsplit

from harness_schema.models import Run, Triple


@dataclass(frozen=True)
class TripleScore:
    """Set based precision/recall for canonical triples."""

    predicted: int
    gold: int
    correct: int

    @property
    def precision(self) -> float:
        return self.correct / self.predicted if self.predicted else 0.0

    @property
    def recall(self) -> float:
        return self.correct / self.gold if self.gold else 0.0

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        return 2 * self.precision * self.recall / denominator if denominator else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {"predicted": self.predicted, "gold": self.gold, "correct": self.correct,
                "precision": self.precision, "recall": self.recall, "f1": self.f1}


PREDICATE_ALIASES = {
    "cites": "https://dblp.org/rdf/schema#cites",
    "authoredby": "https://dblp.org/rdf/schema#authoredBy",
    "title": "https://dblp.org/rdf/schema#title",
    "year": "https://dblp.org/rdf/schema#yearOfPublication",
    "venue": "https://dblp.org/rdf/schema#publishedIn",
}


def _entity_key(value: object, *, predicate: str, position: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if re.fullmatch(r"p\d+", raw, re.IGNORECASE):
        return f"https://harness.local/resource/{raw.lower()}"
    if raw.startswith("hrn:"):
        return raw.removeprefix("hrn:")
    if predicate == PREDICATE_ALIASES["authoredby"] and position == "object":
        if raw.startswith("https://harness.local/resource/author/"):
            name = unquote(raw.rsplit("/", 1)[-1]).replace("_", " ")
        elif raw.startswith("https://harness.local/resource/"):
            name = unquote(raw.rsplit("/", 1)[-1]).replace("%20", " ").replace("_", " ")
        else:
            name = raw
        slug = re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")
        return f"https://harness.local/resource/author/{slug}"
    if raw.startswith("https://"):
        parts = urlsplit(raw)
        return parts._replace(path=parts.path.rstrip("/")).geturl()
    return " ".join(raw.casefold().split())


def _predicate_key(value: object) -> str:
    raw = str(value or "").strip()
    if raw.casefold() in PREDICATE_ALIASES:
        return PREDICATE_ALIASES[raw.casefold()]
    if raw.startswith("hrn:"):
        return PREDICATE_ALIASES.get(raw[4:].casefold(), raw[4:])
    return raw.rstrip("/")


def canonical_triple_key(value: Triple | Mapping[str, object]) -> tuple[str, str, str]:
    if isinstance(value, Triple):
        subject, predicate, object_ = value.s, value.p, value.o
    else:
        subject, predicate, object_ = value.get("s", ""), value.get("p", ""), value.get("o", "")
    predicate_key = _predicate_key(predicate)
    return (
        _entity_key(subject, predicate=predicate_key, position="subject"),
        predicate_key,
        _entity_key(object_, predicate=predicate_key, position="object"),
    )


def score_triples(predicted: Iterable[Triple | Mapping[str, object]], gold: Iterable[Triple | Mapping[str, object]]) -> TripleScore:
    """Score unique triples, making duplicate extraction harmless."""
    predicted_set = {canonical_triple_key(item) for item in predicted}
    gold_set = {canonical_triple_key(item) for item in gold}
    return TripleScore(len(predicted_set), len(gold_set), len(predicted_set & gold_set))


def evaluate_run(run: Run, gold_triples: Iterable[Triple | Mapping[str, object]]) -> dict:
    """Evaluate every reading and the selected reading without mutating the run."""
    gold = list(gold_triples)
    gold_keys = {canonical_triple_key(item) for item in gold}
    rows = []
    for index, reading in enumerate(run.readings):
        score = score_triples(reading.triples, gold)
        predicted_keys = {canonical_triple_key(item) for item in reading.triples}
        supported = sum(bool(item.source.strip() and item.evidence.strip()) for item in reading.triples)
        answer_sources = set(re.findall(r"\[([^\]]+)\]", reading.answer))
        declared_sources = {part.strip() for group in answer_sources for part in group.split(",")}
        triple_sources = {item.source for item in reading.triples}
        rows.append(
            {
                "index": index,
                "label": reading.label,
                "score": score.as_dict(),
                "provenance_coverage": supported / len(reading.triples) if reading.triples else 0.0,
                "unsupported_ratio": 1 - (score.correct / score.predicted if score.predicted else 0.0),
                "answer_supported": declared_sources <= triple_sources if declared_sources else True,
                "missing_gold": len(gold_keys - predicted_keys),
            }
        )
    selected = next((row["score"] for row in rows if row["index"] == run.chosen_reading), None)
    return {"run_id": run.id, "path": run.path, "status": run.status, "gold": len(gold_keys), "readings": rows, "chosen_reading": run.chosen_reading, "chosen": selected}


__all__ = ["TripleScore", "canonical_triple_key", "evaluate_run", "score_triples"]
