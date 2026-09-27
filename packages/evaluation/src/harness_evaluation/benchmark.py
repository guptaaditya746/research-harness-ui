"""Benchmark aggregation: every approved gold row against the runs of each path.

Pure (no API state): the API's ``/benchmark`` and ``make evaluate`` both call
``build_benchmark``. A run is paired with a gold row when the question text and
the path match and, when the row has a snapshot, the run's evaluation snapshot
is identical (same corpus, ontology, claim schema and index versions).
Repetitions are averaged per question first; intervals are over questions.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from statistics import mean

from harness_schema.models import Run

from . import LEVELS, evaluate_gold_row
from .matching import Judge
from .relations import PredicateRelations
from .stats import mean_ci, paired_difference_ci

LOGGER = logging.getLogger(__name__)
PATHS = ("baseline", "baseline-enhanced", "agentic", "external")


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
            for row in members:
                entry = next((item for item in row["paths"] if item["path"] == path), None)
                if not entry or entry["levels"]["soft"]["f1"]["mean"] is None:
                    continue
                for level in LEVELS:
                    scores[level].append(entry["levels"][level]["f1"]["mean"])
                if entry["vital_recall"]["mean"] is not None:
                    vital.append(entry["vital_recall"]["mean"])
                values.setdefault(path, {})[row["id"]] = entry["levels"]["soft"]["f1"]["mean"]
            per_path[path] = {**{level: mean_ci(scores[level]) for level in LEVELS}, "vital_recall": mean_ci(vital)}
        differences = {}
        # Each path against Fast, and ours against the external baseline (paired over shared questions).
        for path, other in [*((path, "baseline") for path in paths[1:]), ("agentic", "external")]:
            shared = sorted(set(values.get(path, {})) & set(values.get(other, {})))
            differences[f"{path} - {other}"] = paired_difference_ci(
                [values[path][qid] for qid in shared], [values[other][qid] for qid in shared])
        rows.append({"complexity": group, "questions": len(members), "paths": per_path,
                     "soft_f1_differences": differences})
    return {"metric": "triple F1 per question, mean and 95% bootstrap interval over questions", "rows": rows}


def matching_runs(candidate: dict, runs: Iterable[Run], path: str) -> list[Run]:
    """The runs of ``path`` for this gold row: same question text and, if the row has one, same snapshot."""
    matching = [run for run in runs if run.question == candidate.get("question") and run.path == path]
    if candidate.get("snapshot") is not None:
        matching = [run for run in matching
                    if run.snapshot and run.snapshot.model_dump(by_alias=True) == candidate["snapshot"]]
    return matching


def build_benchmark(
    candidates: Iterable[dict], runs: Iterable[Run], *, relations: PredicateRelations,
    judge: Judge | None = None, cost: Callable[[Run], dict] | None = None,
) -> dict:
    """Scores per question and path, plus the complexity × path summary."""
    runs = list(runs)
    question_rows = []
    for candidate in candidates:
        path_rows = []
        for path in PATHS:
            matching = matching_runs(candidate, runs, path)
            evaluations = []
            for run in matching:
                if run.status != "answered":
                    continue
                try:
                    evaluations.append(evaluate_gold_row(run, candidate, relations=relations, judge=judge))
                except (ValueError, TypeError, KeyError) as exc:
                    LOGGER.warning(
                        "Cannot score run %s against %s: %s", run.id, candidate.get("id"), exc
                    )
            eligible = [result for result in evaluations if result["eligible_for_path_score"]]
            scores = [result["chosen"] for result in eligible if result["chosen"] is not None]
            claim_scores = [
                reading["claim_score"]["claim"]
                for result in eligible
                for reading in result["readings"]
                if reading["index"] == result["chosen_reading"]
                and reading["claim_score"] is not None
            ]
            level_rows = [result["chosen_levels"] for result in eligible if result["chosen_levels"]]
            costs = [cost(run) for run in matching] if cost else []
            if not matching:
                continue
            path_rows.append(
                {
                    "path": "adaptive-baseline" if path == "baseline-enhanced" else path,
                    "repetitions": [run.id for run in matching],
                    "completed": len(scores),
                    "failed": len(matching) - len(scores),
                    "precision": _score_aggregate(scores, "precision"),
                    "recall": _score_aggregate(scores, "recall"),
                    "f1": _score_aggregate(scores, "f1"),
                    "claim_precision": _score_aggregate(claim_scores, "precision"),
                    "claim_recall": _score_aggregate(claim_scores, "recall"),
                    "claim_f1": _score_aggregate(claim_scores, "f1"),
                    "levels": {
                        level: {
                            metric: _score_aggregate([row["levels"][level] for row in level_rows], metric)
                            for metric in ("precision", "recall", "f1")
                        }
                        for level in LEVELS
                    },
                    "vital_recall": _value_aggregate(
                        [row["levels"]["soft"]["vital_recall"] for row in level_rows]),
                    "judged_precision": _value_aggregate(
                        [row["judged_precision"]["value"] for row in level_rows]),
                    "judge_pending": sum(row["judge"]["pending"] for row in level_rows),
                    "accepted_readings": sorted({
                        result["accepted_reading"]["label"] for result in eligible
                        if result.get("accepted_reading")}),
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
    }


__all__ = ["PATHS", "build_benchmark", "complexity_level", "matching_runs"]
