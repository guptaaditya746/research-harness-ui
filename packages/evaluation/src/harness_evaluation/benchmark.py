"""Benchmark aggregation: every approved gold row against the runs of each path.

Pure (no API state): the API's ``/benchmark`` and ``make evaluate`` both call
``build_benchmark``. A run is paired with a gold row when the question text and
the path match, the run read the chosen input (``run_input``: "corpus", or
"evidence_doc" when the evidence document D was its input; the two are never
mixed) and, when the row has a snapshot, the run's evaluation snapshot is
identical (same corpus, ontology, claim schema and index versions).

The scored unit is an **attempt**: the runs sharing ``Run.attempt_id`` (an Agentic attempt researches
each proposed reading of an ambiguous question in its own run) form one prediction, the union of their
chosen readings' triples, scored with AmbigQA set F1 against the union of the row's accepted readings
(``evaluate_attempt``); older reading runs without an attempt id are grouped by their proposal
(``parent_run_id``), any other run is an attempt of its own (``group_attempts``). Reading runs are never
averaged as repetitions. Repetitions (attempts) are averaged per question first; intervals are over
questions.

Every attempt counts. An attempt that ended without a scorable answer (no run of it
answered: incomplete, no evidence, corpus absent, degraded, or produced by another path) is scored as an
empty prediction: precision, recall and F1 0, vital recall 0. This is how WebNLG,
Text2KGBench and TREC treat a missing output; averaging only the answered runs would
reward a path for failing on the hard questions (and drop those questions from the
paired comparison). The answered-only means are kept as a secondary view
(``answered_only``) with their ``coverage`` (answered / attempted). Runs waiting for
a reading choice or stopped by a person are not attempts (``not_attempted``). Rows
whose ``expected_status`` is not "answered" keep the answered-only scoring: an
unanswerable question has no gold to miss.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from statistics import mean

from harness_schema.models import Run

from . import LEVELS, accepted_readings, evaluate_attempt, reading_labels, row_gold
from .matching import Judge
from .relations import PredicateRelations
from .stats import mean_ci, paired_difference_ci

LOGGER = logging.getLogger(__name__)
PATHS = ("baseline", "baseline-enhanced", "agentic", "external")
# Unfinished (waiting for a person to pick a reading) or stopped by a person: not an attempt.
NOT_ATTEMPTS = {"awaiting_reading", "stopped"}
ZERO = {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def _score_aggregate(scores: list[dict], metric: str) -> dict[str, float | None]:
    values = [score[metric] for score in scores]
    return {
        "mean": mean(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def _value_aggregate(values: list[float | None]) -> dict[str, float | None]:
    present = [value for value in values if value is not None]
    return {
        "mean": mean(present) if present else None,
        "min": min(present) if present else None,
        "max": max(present) if present else None,
    }


def _empty_prediction(candidate: dict) -> dict:
    """What an attempt without a scorable answer scores: an empty triple set against the row's gold."""
    gold = row_gold(candidate)
    vital = 0.0 if any(isinstance(item, dict) and item.get("weight") == "vital" for item in gold) else None
    readings = accepted_readings(candidate)
    return {"chosen": dict(ZERO), "claim": dict(ZERO) if candidate.get("gold_claims") is not None else None,
            "levels": {"levels": {level: {**ZERO, "vital_recall": vital} for level in LEVELS},
                       "judged_precision": {"value": None}, "judge": {"pending": 0}},
            "readings_recall": {label: 0.0 for label in reading_labels(candidate)},
            "readings_covered": 0 if readings else None}


def group_attempts(runs: Iterable[Run]) -> list[list[Run]]:
    """Runs grouped into attempts, in first-seen order. The key is ``attempt_id``; else ``parent_run_id``
    (the reading runs of one proposal, as ``make run-questions`` recorded them before attempt ids existed;
    a person's pick in the UI starts one run per proposal, so it stays alone); else the run's own id. So a
    run without either is its own attempt, and the run whose id the others share joins them."""
    groups: dict[str, list[Run]] = {}
    for run in runs:
        groups.setdefault(run.attempt_id or run.parent_run_id or run.id, []).append(run)
    return list(groups.values())


def _readings_recall_mean(entries: list[dict]) -> dict[str, float]:
    labels = list(dict.fromkeys(label for entry in entries for label in entry.get("readings_recall") or {}))
    return {label: mean(entry["readings_recall"].get(label, 0.0) for entry in entries
                        if entry.get("readings_recall") is not None) for label in labels}


def _level_summary(level_rows: list[dict]) -> dict:
    return {
        "levels": {level: {metric: _score_aggregate([row["levels"][level] for row in level_rows], metric)
                           for metric in ("precision", "recall", "f1")} for level in LEVELS},
        "vital_recall": _value_aggregate([row["levels"]["soft"]["vital_recall"] for row in level_rows]),
    }


def complexity_level(candidate: dict) -> str | None:
    complexity = candidate.get("complexity")
    level = complexity.get("level") if isinstance(complexity, dict) else None
    return level or candidate.get("difficulty")


def _by_complexity(question_rows: list[dict]) -> dict:
    """Per-question means (repetitions averaged first), then a bootstrap interval over questions."""
    groups = ["simple", "compound", "complex"]
    paths = ["baseline", "adaptive-baseline", "agentic", "external"]
    rows = []
    for group in [*groups, "all"]:
        members = [row for row in question_rows if group == "all" or row.get("complexity") == group]
        per_path: dict[str, dict] = {}
        values: dict[str, dict[str, float]] = {}
        for path in paths:
            scores: dict[str, list[float]] = {level: [] for level in LEVELS}
            vital: list[float] = []
            coverage: list[float] = []
            answered_only: list[float] = []
            for row in members:
                entry = next((item for item in row["paths"] if item["path"] == path), None)
                if not entry or entry["levels"]["soft"]["f1"]["mean"] is None:
                    continue
                for level in LEVELS:
                    scores[level].append(entry["levels"][level]["f1"]["mean"])
                if entry["vital_recall"]["mean"] is not None:
                    vital.append(entry["vital_recall"]["mean"])
                if entry.get("coverage") is not None:
                    coverage.append(entry["coverage"])
                only = entry.get("answered_only", {}).get("levels", {}).get("soft", {}).get("f1", {}).get("mean")
                if only is not None:
                    answered_only.append(only)
                values.setdefault(path, {})[row["id"]] = entry["levels"]["soft"]["f1"]["mean"]
            per_path[path] = {**{level: mean_ci(scores[level]) for level in LEVELS}, "vital_recall": mean_ci(vital),
                              "coverage": mean_ci(coverage), "soft_answered_only": mean_ci(answered_only)}
        differences = {}
        # Each path against Fast, and ours against the external baseline (paired over shared questions).
        for path, other in [*((path, "baseline") for path in paths[1:]), ("agentic", "external")]:
            shared = sorted(set(values.get(path, {})) & set(values.get(other, {})))
            differences[f"{path} - {other}"] = paired_difference_ci(
                [values[path][qid] for qid in shared], [values[other][qid] for qid in shared])
        rows.append({"complexity": group, "questions": len(members), "paths": per_path,
                     "soft_f1_differences": differences})
    return {"metric": ("triple F1 per question (a failed attempt scores 0), mean and 95% BCa bootstrap interval "
                       "over questions; differences paired over shared questions with a sign-flip permutation "
                       "p-value"), "rows": rows}


def matching_runs(candidate: dict, runs: Iterable[Run], path: str, run_input: str = "corpus") -> list[Run]:
    """The runs of ``path`` for this gold row: same question text, same input and, if the row has one, same
    snapshot."""
    matching = [run for run in runs if run.question == candidate.get("question") and run.path == path
                and run.run_input == run_input]
    if candidate.get("snapshot") is not None:
        matching = [run for run in matching
                    if run.snapshot and run.snapshot.model_dump(by_alias=True) == candidate["snapshot"]]
    return matching


def build_benchmark(
    candidates: Iterable[dict], runs: Iterable[Run], *, relations: PredicateRelations,
    judge: Judge | None = None, cost: Callable[[Run], dict] | None = None, run_input: str = "corpus",
) -> dict:
    """Scores per question and path, plus the complexity × path summary, for runs of one input."""
    runs = list(runs)
    question_rows = []
    for candidate in candidates:
        path_rows = []
        for path in PATHS:
            matching = matching_runs(candidate, runs, path, run_input)
            attempts = group_attempts(run for run in matching if run.status not in NOT_ATTEMPTS)
            zero_fill = candidate.get("expected_status", "answered") == "answered"
            scored, counted = [], []           # answered-only; every attempt (failures as empty predictions)
            unscorable: list[str] = []
            unscorable_attempts = 0
            for group in attempts:
                result = None
                if any(run.status == "answered" for run in group):
                    try:
                        result = evaluate_attempt(group, candidate, relations=relations, judge=judge)
                    except (ValueError, TypeError, KeyError) as exc:
                        LOGGER.warning("Cannot score runs %s against %s: %s", [run.id for run in group],
                                       candidate.get("id"), exc)
                        unscorable.extend(run.id for run in group)   # a scoring error is not the path's failure
                        unscorable_attempts += 1
                        continue
                if result is not None and result["eligible_for_path_score"]:
                    entry = {"chosen": result["chosen"], "claim": result["claim"], "levels": result["chosen_levels"],
                             "readings_recall": result["readings_recall"],
                             "readings_covered": result["readings_covered"]}
                    scored.append(entry)
                    counted.append(entry)
                elif zero_fill:
                    counted.append(_empty_prediction(candidate))
            scores = [entry["chosen"] for entry in counted]
            claim_scores = [entry["claim"] for entry in counted if entry["claim"] is not None]
            level_rows = [entry["levels"] for entry in counted if entry["levels"]]
            answered_rows = [entry["levels"] for entry in scored if entry["levels"]]
            costs = [cost(run) for run in matching] if cost else []
            if not matching:
                continue
            path_rows.append(
                {
                    "path": "adaptive-baseline" if path == "baseline-enhanced" else path,
                    "repetitions": [run.id for run in matching],
                    # The runs of each attempt (several when an attempt researched several readings).
                    "attempts": [[run.id for run in group] for group in attempts],
                    "attempted": len(attempts) - unscorable_attempts,
                    "completed": len(scored),
                    "failed": len(attempts) - unscorable_attempts - len(scored),
                    "coverage": len(scored) / (len(attempts) - unscorable_attempts)
                    if len(attempts) > unscorable_attempts else None,
                    "unscorable": unscorable,
                    "not_attempted": [run.id for run in matching if run.status in NOT_ATTEMPTS],
                    "precision": _score_aggregate(scores, "precision"),
                    "recall": _score_aggregate(scores, "recall"),
                    "f1": _score_aggregate(scores, "f1"),
                    "claim_precision": _score_aggregate(claim_scores, "precision"),
                    "claim_recall": _score_aggregate(claim_scores, "recall"),
                    "claim_f1": _score_aggregate(claim_scores, "f1"),
                    **_level_summary(level_rows),
                    "answered_only": {**_level_summary(answered_rows),
                                      "f1": _score_aggregate([entry["chosen"] for entry in scored], "f1")},
                    "judged_precision": _value_aggregate(
                        [row["judged_precision"]["value"] for row in answered_rows]),
                    "judge_pending": sum(row["judge"]["pending"] for row in answered_rows),
                    # Ambiguous rows: mean soft recall per accepted reading and readings covered, per attempt
                    # (failed attempts count as 0, like the main scores).
                    "readings_recall": _readings_recall_mean(counted),
                    "readings_covered": _value_aggregate([entry.get("readings_covered") for entry in counted]),
                    "model_calls": mean([item["model_calls"] for item in costs]) if costs else None,
                    "tool_calls": mean([item["tool_calls"] for item in costs]) if costs else None,
                    "wall_clock_s": mean([item["wall_clock_s"] for item in costs]) if costs else None,
                }
            )
        question_rows.append(
            {
                "id": candidate.get("id"),
                "question": candidate.get("question"),
                "category": candidate.get("category", "uncategorized"),
                "theme": candidate.get("theme"),
                "difficulty": candidate.get("difficulty"),
                "complexity": complexity_level(candidate),
                "complexity_features": (candidate.get("complexity") or {}).get("features")
                if isinstance(candidate.get("complexity"), dict) else None,
                "expected_status": candidate.get("expected_status", "answered"),
                "gold_version": candidate.get("gold_version"),
                "corpus_fingerprint": candidate.get("corpus_fingerprint"),
                "ontology_hash": candidate.get("ontology_hash"),
                "snapshot": candidate.get("snapshot"),
                "paths": path_rows,
            }
        )
    return {
        "questions": question_rows,
        "categories": sorted({row["category"] for row in question_rows}),
        "by_complexity": _by_complexity(question_rows),
        "relations": list(relations.source_files),
        "run_input": run_input,
    }


__all__ = ["NOT_ATTEMPTS", "PATHS", "build_benchmark", "complexity_level", "group_attempts", "matching_runs"]
