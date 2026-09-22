"""Pure artifact evaluation for research runs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

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


def _triple_key(value: Triple | Mapping[str, object]) -> tuple[str, str, str]:
    if isinstance(value, Triple):
        return value.s, value.p, value.o
    return str(value.get("s", "")), str(value.get("p", "")), str(value.get("o", ""))


def score_triples(predicted: Iterable[Triple | Mapping[str, object]], gold: Iterable[Triple | Mapping[str, object]]) -> TripleScore:
    """Score unique triples, making duplicate extraction harmless."""
    predicted_set = {_triple_key(item) for item in predicted}
    gold_set = {_triple_key(item) for item in gold}
    return TripleScore(len(predicted_set), len(gold_set), len(predicted_set & gold_set))


def evaluate_run(run: Run, gold_triples: Iterable[Triple | Mapping[str, object]]) -> dict:
    """Evaluate every reading and the selected reading without mutating the run."""
    gold = list(gold_triples)
    rows = [{"index": index, "label": reading.label, "score": score_triples(reading.triples, gold).as_dict()} for index, reading in enumerate(run.readings)]
    selected = next((row["score"] for row in rows if row["index"] == run.chosen_reading), None)
    return {"run_id": run.id, "path": run.path, "status": run.status, "gold": len({_triple_key(item) for item in gold}), "readings": rows, "chosen_reading": run.chosen_reading, "chosen": selected}


__all__ = ["TripleScore", "evaluate_run", "score_triples"]
