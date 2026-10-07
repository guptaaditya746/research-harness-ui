"""Evaluation of saved runs against gold.

- ``canonical_triple_key`` / ``score_triples``: exact set-based triple scores (the original metric).
- ``matching.score_levels``: precision and recall at the strict, entailed and soft levels, one-to-one,
  with locally closed DBLP scoring, vital recall and judged precision.
- ``relations``: RDFS/OWL rules for the entailed level (closed with owlrl).
- ``judge``: the cached soft-match judge (no model calls unless the caller passes one).
- ``stats``: bootstrap confidence intervals over questions.
- ``evaluate_gold_row`` / ``evaluate_attempt``: a run, or the runs of one attempt, against one gold row;
  an ambiguous question is scored AmbigQA-style (set F1 against the union of its accepted readings).

The metric definitions, their reasons and their limits are documented in the
labelling repository (``docs/EVALUATION.md``).
"""

from __future__ import annotations

import importlib.util
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import unquote, urlsplit

import numpy as np
from harness_config import settings
from harness_schema.models import ClaimSubgraph, Run, Triple
from scipy.optimize import linear_sum_assignment

from .matching import LEVELS, Judge, LevelScore, score_levels
from .relations import PredicateRelations, load_relations


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
    "author": "https://dblp.org/rdf/schema#authoredBy",
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
        if parts.netloc.casefold() == "dblp.org":
            return parts._replace(path=parts.path.rstrip("/").casefold()).geturl()
        return parts._replace(path=parts.path.rstrip("/")).geturl()
    return " ".join(raw.casefold().split())


def _predicate_key(value: object) -> str:
    raw = str(value or "").strip()
    if raw.casefold() in PREDICATE_ALIASES:
        return PREDICATE_ALIASES[raw.casefold()]
    if raw.startswith("hrn:"):
        return PREDICATE_ALIASES.get(raw[4:].casefold(), raw[4:])
    return raw.rstrip("/")


@lru_cache(maxsize=1)
def _shared_vocabulary() -> tuple[object, dict] | None:
    normalizer_path = settings().labelling_normalization_file
    vocabulary_path = settings().labelling_vocabulary_file
    if not normalizer_path or not vocabulary_path:
        return None
    module_spec = importlib.util.spec_from_file_location("labelling_shared_normalization", normalizer_path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"cannot load shared normalization module: {normalizer_path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module, json.loads(Path(vocabulary_path).read_text())


def canonical_triple_key(
    value: Triple | Mapping[str, object], question_id: str | None = None,
) -> tuple[str, str, str]:
    if isinstance(value, Triple):
        subject, predicate, object_ = value.s, value.p, value.o
    else:
        subject, predicate, object_ = value.get("s", ""), value.get("p", ""), value.get("o", "")
    shared = _shared_vocabulary()
    if question_id and shared:
        module, vocabulary = shared
        if question_id in vocabulary.get("question_texts", {}):
            if vocabulary.get("status") != "frozen":
                raise RuntimeError("freeze the corpus_10 answer vocabulary before scoring")
            return module.canonical_triple_key(
                {"s": subject, "p": predicate, "o": object_}, question_id, vocabulary
            )
    elif question_id and settings().labelling_vocabulary_file:
        raise RuntimeError("shared corpus_10 normalization could not be loaded")
    predicate_key = _predicate_key(predicate)
    return (
        _entity_key(subject, predicate=predicate_key, position="subject"),
        predicate_key,
        _entity_key(object_, predicate=predicate_key, position="object"),
    )


def score_triples(
    predicted: Iterable[Triple | Mapping[str, object]],
    gold: Iterable[Triple | Mapping[str, object]],
    *, question_id: str | None = None,
) -> TripleScore:
    """Score unique triples, making duplicate extraction harmless."""
    predicted_set = {canonical_triple_key(item, question_id) for item in predicted}
    gold_set = {canonical_triple_key(item, question_id) for item in gold}
    return TripleScore(len(predicted_set), len(gold_set), len(predicted_set & gold_set))


def _claim_fields(claim: ClaimSubgraph) -> dict[str, str]:
    return {
        key: " ".join(field.value.casefold().split())
        for key, field in claim.fields.items()
        if key != "relation"
    }


def score_claims(
    predicted: Iterable[ClaimSubgraph | Mapping[str, object]],
    gold: Iterable[ClaimSubgraph | Mapping[str, object]],
) -> dict:
    """Align attributed claims, then expose complete and per-field correctness."""
    predictions = [item if isinstance(item, ClaimSubgraph) else ClaimSubgraph.model_validate(item)
                   for item in predicted]
    targets = [item if isinstance(item, ClaimSubgraph) else ClaimSubgraph.model_validate(item)
               for item in gold]
    # Optimal alignment (as for triples): the one-to-one pairing with the most equal fields in total,
    # then the most claims equal in every field. A greedy pass can lose fields: it may give a claim to a
    # slightly better partner that another claim needed.
    common = np.zeros((len(predictions), len(targets)))
    exact = np.zeros_like(common)
    for pi, prediction in enumerate(predictions):
        for gi, target in enumerate(targets):
            if prediction.kind != target.kind or prediction.paper_id != target.paper_id:
                continue
            predicted_fields, gold_fields = _claim_fields(prediction), _claim_fields(target)
            common[pi, gi] = sum(predicted_fields.get(key) == value for key, value in gold_fields.items()
                                 if key in predicted_fields)
            exact[pi, gi] = float(predicted_fields == gold_fields)
    matches: list[tuple[int, int]] = []
    if common.size and common.any():
        size = min(common.shape) + 1                   # the exact-claim bonus never outweighs one field
        rows, cols = linear_sum_assignment(common * size + exact * (common > 0), maximize=True)
        matches = sorted((int(pi), int(gi)) for pi, gi in zip(rows, cols, strict=True) if common[pi, gi] > 0)
    correct_claims = 0
    correct_fields = 0
    details = []
    for pi, gi in matches:
        left, right = _claim_fields(predictions[pi]), _claim_fields(targets[gi])
        exact = left == right
        correct_claims += int(exact)
        matched = sorted(key for key in left.keys() & right.keys() if left[key] == right[key])
        correct_fields += len(matched)
        details.append({
            "predicted_id": predictions[pi].id, "gold_id": targets[gi].id,
            "exact": exact, "matched_fields": matched,
            "wrong_or_extra_fields": sorted(key for key in left if key not in matched),
            "missing_fields": sorted(key for key in right if key not in matched),
        })
    predicted_fields = sum(len(_claim_fields(item)) for item in predictions)
    gold_fields = sum(len(_claim_fields(item)) for item in targets)
    claim_score = TripleScore(len(predictions), len(targets), correct_claims)
    field_score = TripleScore(predicted_fields, gold_fields, correct_fields)
    return {"claim": claim_score.as_dict(), "fields": field_score.as_dict(),
            "aligned": len(matches), "details": details}


def _question_id(question: str, question_id: str | None) -> str | None:
    """The question id the shared corpus_10 normalization needs (looked up by text when not given)."""
    shared = _shared_vocabulary()
    if question_id is None and shared:
        module, vocabulary = shared
        return module.question_id_for_text(question, vocabulary)
    return question_id


def eligible_for_path_score(run: Run) -> bool:
    """An answered, non-degraded run produced by its own path (anything else scores as an empty prediction)."""
    return (run.status == "answered" and run.execution_status != "degraded"
            and not (run.reason or "").startswith("agentic_degraded:")
            and run.producing_path in {None, run.path})


def evaluate_run(
    run: Run, gold_triples: Iterable[Triple | Mapping[str, object]],
    gold_claims: Iterable[ClaimSubgraph | Mapping[str, object]] | None = None,
    *, question_id: str | None = None, relations: PredicateRelations | None = None,
    judge: Judge | None = None, rejected: Iterable[Mapping[str, object]] = (),
) -> dict:
    """Evaluate every reading and the selected reading without mutating the run.

    ``score``/``chosen`` keep the original exact set-based metric; ``levels``/``chosen_levels``
    add the strict/entailed/soft scores, vital recall, judged precision and the matched,
    missed and extra triples for the Interpretation window.
    """
    gold = list(gold_triples)
    rejected = list(rejected)
    relations = relations if relations is not None else load_relations()
    claim_gold = list(gold_claims) if gold_claims is not None else None
    question_id = _question_id(run.question, question_id)
    gold_keys = {canonical_triple_key(item, question_id) for item in gold}
    rows = []
    for index, reading in enumerate(run.readings):
        score = score_triples(reading.triples, gold, question_id=question_id)
        predicted_keys = {canonical_triple_key(item, question_id) for item in reading.triples}
        supported = sum(bool(item.source.strip() and item.evidence.strip()) for item in reading.triples)
        answer_sources = set(re.findall(r"\[([^\]]+)\]", reading.answer))
        declared_sources = {part.strip() for group in answer_sources for part in group.split(",")}
        triple_sources = {item.source for item in reading.triples}
        claim_ids = {claim.id for claim in reading.claims}
        answer_claim_ids = set(reading.answer_claim_ids)
        claim_grounded = (not reading.answer.strip() or bool(answer_claim_ids)) and answer_claim_ids <= claim_ids
        rows.append(
            {
                "index": index,
                "label": reading.label,
                "score": score.as_dict(),
                "provenance_coverage": supported / len(reading.triples) if reading.triples else 0.0,
                # Share of the reading's triples that are not in the gold (1 - exact precision). Not a
                # provenance measure: provenance_coverage says how many carry a source and evidence.
                "not_in_gold_ratio": 1 - (score.correct / score.predicted if score.predicted else 0.0),
                # New runs declare exactly which accepted/partial source claims feed
                # the prose. Older artifacts retain the citation-only fallback.
                "answer_supported": claim_grounded and (
                    declared_sources <= triple_sources if declared_sources else True
                ),
                "answer_claim_ids": sorted(answer_claim_ids),
                "missing_gold": len(gold_keys - predicted_keys),
                "claim_score": score_claims(reading.claims, claim_gold) if claim_gold is not None else None,
                # The judge is only consulted for the chosen reading, the one that is reported.
                "levels": score_levels(
                    reading.triples, gold, key=lambda item: canonical_triple_key(item, question_id),
                    relations=relations, judge=judge if index == run.chosen_reading else None,
                    rejected=rejected,
                ),
            }
        )
    selected = next((row["score"] for row in rows if row["index"] == run.chosen_reading), None)
    selected_levels = next((row["levels"] for row in rows if row["index"] == run.chosen_reading), None)
    return {"run_id": run.id, "path": run.path, "status": run.status,
            "producing_path": run.producing_path, "execution_status": run.execution_status,
            "eligible_for_path_score": eligible_for_path_score(run),
            "gold": len(gold_keys), "gold_claims": len(claim_gold) if claim_gold is not None else None,
            "readings": rows, "chosen_reading": run.chosen_reading, "chosen": selected,
            "chosen_levels": selected_levels}


def accepted_readings(row: Mapping[str, object]) -> list[Mapping[str, object]]:
    """The row's accepted readings that have gold triples. A reading without computable gold cannot be
    scored or covered."""
    return [item for item in row.get("accepted_readings") or [] if item.get("gold_triples")]


def reading_labels(row: Mapping[str, object]) -> list[str]:
    """One distinct label per accepted reading (its label, else its id, else its position)."""
    labels: list[str] = []
    for index, reading in enumerate(accepted_readings(row)):
        label = str(reading.get("label") or reading.get("reading_id") or f"reading {index + 1}")
        labels.append(f"{label} ({index + 1})" if label in labels else label)
    return labels


def row_gold(row: Mapping[str, object]) -> list:
    """The gold a prediction is scored against: the row's ``gold_triples`` plus every accepted reading's.

    AmbigQA (Min et al., 2020) scores a prediction, a set of (interpretation, answer) pairs, with F1
    against all interpretations of the question; the union is that set here. In labelling's export the
    row's own ``gold_triples`` are the triples reviewed without an interpretation (empty for an ambiguous
    question) and each reading carries only the triples reviewed under it, so the union counts nothing
    twice. A triple in several of them (same canonical key) is one gold fact: the scorer merges the copies,
    vital if any copy is vital, their variants pooled.
    """
    return [*(row.get("gold_triples") or []),
            *(item for reading in accepted_readings(row) for item in reading["gold_triples"] or [])]


def _readings_recall(
    triples: list, row: Mapping[str, object], *, question_id: str | None,
    relations: PredicateRelations, judge: Judge | None,
) -> dict[str, float]:
    """Soft recall of a prediction against each accepted reading's own gold, by reading label."""
    recall: dict[str, float] = {}
    for label, reading in zip(reading_labels(row), accepted_readings(row), strict=True):
        levels = score_levels(triples, reading["gold_triples"],
                              key=lambda item: canonical_triple_key(item, question_id),
                              relations=relations, judge=judge)
        recall[label] = float(levels["levels"]["soft"]["recall"])
    return recall


def _readings_summary(recall: dict[str, float], row: Mapping[str, object]) -> dict:
    readings = accepted_readings(row)
    return {"levels_gold": "readings_union" if readings else "row",
            "readings_recall": recall,
            "readings_covered": sum(value > 0 for value in recall.values()) if readings else None,
            "readings_accepted": len(readings)}


def evaluate_gold_row(
    run: Run, row: Mapping[str, object], *, relations: PredicateRelations | None = None,
    judge: Judge | None = None, question_id: str | None = None,
) -> dict:
    """Evaluate one run against one gold row, using everything the row carries.

    Reviewer-rejected triples (``forbidden_alternatives``) feed judged precision. For an ambiguous question
    the gold is the union of the accepted readings' gold (``row_gold``; ``levels_gold`` = "readings_union"),
    scored once, AmbigQA set F1: a run triple is correct if it matches any reading's gold, and recall is over
    all readings, so a run that answers one of two equally large readings gets recall of about 1/2. The
    exact metric (``chosen``) uses the same union. ``readings_recall`` gives the chosen reading's soft recall
    against each reading's own gold, ``readings_covered`` how many readings it recovers anything of. A run
    researches one reading; the runs of one attempt are scored together by ``evaluate_attempt``.
    """
    relations = relations if relations is not None else load_relations()
    question_id = _question_id(run.question, question_id)
    result = evaluate_run(run, row_gold(row), row.get("gold_claims"), question_id=question_id,
                          relations=relations, judge=judge,
                          rejected=list(row.get("forbidden_alternatives") or []))
    chosen = run.chosen_reading
    triples = list(run.readings[chosen].triples) if chosen is not None and chosen < len(run.readings) else []
    recall = _readings_recall(triples, row, question_id=question_id, relations=relations,
                              judge=judge if triples else None) if accepted_readings(row) else {}
    result.update(_readings_summary(recall, row))
    return result


def _reading_label(run: Run) -> str:
    if run.concept is not None:
        return run.concept.label
    chosen = run.chosen_reading
    return run.readings[chosen].label if chosen is not None and chosen < len(run.readings) else run.id


def evaluate_attempt(
    runs: Iterable[Run], row: Mapping[str, object], *, relations: PredicateRelations | None = None,
    judge: Judge | None = None, question_id: str | None = None,
) -> dict:
    """Score one attempt, the runs sharing an attempt id (``Run.attempt_id``; a run without one is an
    attempt of its own), as ONE prediction against one gold row.

    The prediction is the union of the chosen-reading triples of the attempt's eligible runs (answered,
    not degraded, produced by their own path); a failed reading run adds nothing, so it lowers recall. A
    reading researched twice in one attempt (a resumed batch) counts once: its latest eligible run.
    The prediction is scored like one run (``evaluate_gold_row``): set F1 against ``row_gold``, plus the
    per-reading recall. An attempt without an eligible run is not ``eligible_for_path_score`` (the
    benchmark scores it as an empty prediction).
    """
    runs = list(runs)
    if not runs:
        raise ValueError("an attempt needs at least one run")
    relations = relations if relations is not None else load_relations()
    question_id = _question_id(runs[0].question, question_id)
    latest: dict[str, Run] = {}
    for run in sorted((run for run in runs if eligible_for_path_score(run)), key=lambda run: run.started_at):
        latest[_reading_label(run)] = run
    used = list(latest.values())
    readings = [run.readings[run.chosen_reading] for run in used if run.chosen_reading is not None]
    triples = [item for reading in readings for item in reading.triples]
    claims = list({claim.id: claim for reading in readings for claim in reading.claims}.values())
    gold = row_gold(row)
    claim_gold = row.get("gold_claims")
    key = lambda item: canonical_triple_key(item, question_id)
    recall = _readings_recall(triples, row, question_id=question_id, relations=relations,
                              judge=judge if triples else None) if accepted_readings(row) else {}
    return {
        "run_ids": [run.id for run in runs],
        "scored_runs": [run.id for run in used],
        "eligible_for_path_score": bool(used),
        "gold": len({key(item) for item in gold}),
        "chosen": score_triples(triples, gold, question_id=question_id).as_dict(),
        "claim": score_claims(claims, claim_gold)["claim"] if claim_gold is not None else None,
        "chosen_levels": score_levels(triples, gold, key=key, relations=relations,
                                      judge=judge if used else None,
                                      rejected=list(row.get("forbidden_alternatives") or [])),
        **_readings_summary(recall, row),
    }


__all__ = [
    "LEVELS", "Judge", "LevelScore", "PredicateRelations", "TripleScore", "accepted_readings",
    "canonical_triple_key", "eligible_for_path_score", "evaluate_attempt", "evaluate_gold_row", "evaluate_run",
    "load_relations", "reading_labels", "row_gold", "score_claims", "score_levels", "score_triples",
]
