"""Evaluation of saved runs against gold.

- ``canonical_triple_key`` / ``score_triples``: exact set-based triple scores (the original metric).
- ``matching.score_levels``: precision and recall at the strict, entailed and soft levels, one-to-one,
  with locally closed DBLP scoring, vital recall and judged precision.
- ``relations``: RDFS/OWL rules for the entailed level (closed with owlrl).
- ``judge``: the cached soft-match judge (no model calls unless the caller passes one).
- ``stats``: bootstrap confidence intervals over questions.

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

from harness_config import settings
from harness_schema.models import ClaimSubgraph, Run, Triple

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
    candidate_pairs = []
    for pi, prediction in enumerate(predictions):
        for gi, target in enumerate(targets):
            if prediction.kind != target.kind or prediction.paper_id != target.paper_id:
                continue
            predicted_fields, gold_fields = _claim_fields(prediction), _claim_fields(target)
            common = sum(predicted_fields.get(key) == value for key, value in gold_fields.items()
                         if key in predicted_fields)
            if common:
                candidate_pairs.append((common, pi, gi))
    matches: list[tuple[int, int]] = []
    used_predictions: set[int] = set()
    used_targets: set[int] = set()
    for _common, pi, gi in sorted(candidate_pairs, key=lambda row: (-row[0], row[1], row[2])):
        if pi not in used_predictions and gi not in used_targets:
            matches.append((pi, gi))
            used_predictions.add(pi)
            used_targets.add(gi)
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
    shared = _shared_vocabulary()
    if question_id is None and shared:
        module, vocabulary = shared
        question_id = module.question_id_for_text(run.question, vocabulary)
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
                "unsupported_ratio": 1 - (score.correct / score.predicted if score.predicted else 0.0),
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
            "eligible_for_path_score": run.status == "answered" and run.execution_status != "degraded" and
            not (run.reason or "").startswith("agentic_degraded:") and
            run.producing_path in {None, run.path},
            "gold": len(gold_keys), "gold_claims": len(claim_gold) if claim_gold is not None else None,
            "readings": rows, "chosen_reading": run.chosen_reading, "chosen": selected,
            "chosen_levels": selected_levels}


def evaluate_gold_row(
    run: Run, row: Mapping[str, object], *, relations: PredicateRelations | None = None,
    judge: Judge | None = None, question_id: str | None = None,
) -> dict:
    """Evaluate a run against one gold row, using everything the row carries.

    Reviewer-rejected triples (``forbidden_alternatives``) feed judged precision. For an ambiguous
    question, ``accepted_readings`` may give one gold set per accepted reading: the run is scored
    against the row's own gold and against each reading's gold, and the best soft F1 is kept (the
    AmbigQA/ASQA convention). ``accepted_reading`` names the reading credited (None = the row's own
    gold). The original exact metric (``chosen``) always refers to the row's own gold.
    """
    relations = relations if relations is not None else load_relations()
    rejected = list(row.get("forbidden_alternatives") or [])
    base = evaluate_run(run, row.get("gold_triples") or [], row.get("gold_claims"), question_id=question_id,
                        relations=relations, judge=judge, rejected=rejected)
    base["accepted_reading"] = None
    readings = [item for item in row.get("accepted_readings") or [] if item.get("gold_triples")]
    if not readings:
        return base

    def soft_f1(result: dict) -> float:
        levels = result.get("chosen_levels") or {}
        return float(levels.get("levels", {}).get("soft", {}).get("f1", 0.0))

    best_score, best_index, best = soft_f1(base), None, base
    for index, reading in enumerate(readings):
        result = evaluate_run(run, reading["gold_triples"], None, question_id=question_id,
                              relations=relations, judge=judge, rejected=rejected)
        if soft_f1(result) > best_score:
            best_score, best_index, best = soft_f1(result), index, result
    if best_index is None:
        return base
    base["chosen_levels"] = best["chosen_levels"]
    for mine, theirs in zip(base["readings"], best["readings"], strict=True):
        mine["levels"] = theirs["levels"]
    base["accepted_reading"] = {"index": best_index, "label": readings[best_index].get("label"),
                                "measure": readings[best_index].get("measure"), "gold": best["gold"]}
    return base


__all__ = [
    "LEVELS", "Judge", "LevelScore", "PredicateRelations", "TripleScore", "canonical_triple_key",
    "evaluate_gold_row", "evaluate_run", "load_relations", "score_claims", "score_levels", "score_triples",
]
