"""Benchmark endpoints: scores per path against approved gold, and evaluation repetitions."""

from __future__ import annotations

import logging
from statistics import mean

import app_state
from api_models import BenchmarkExecuteRequest, RunRequest
from fastapi import APIRouter, HTTPException
from harness_config import settings
from harness_evaluation import evaluate_run
from harness_schema.models import EvaluationSnapshot, Run

router = APIRouter()
LOGGER = logging.getLogger(__name__)


def _score_aggregate(scores: list[dict], metric: str) -> dict[str, float | None]:
    values = [score[metric] for score in scores]
    return {
        "mean": mean(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def _run_cost(run: Run) -> dict[str, float | int]:
    """Return observable per-run costs; tool counts come from the saved call log."""
    model_calls = sum(row.action == "Model call started" for row in run.trace)
    try:
        tool_log = app_state.runtime.artifacts.read(run.id, "tool_calls.jsonl")
        tool_calls = sum(bool(line.strip()) for line in tool_log.splitlines())
    except (OSError, ValueError):
        tool_calls = 0
    return {"model_calls": model_calls, "tool_calls": tool_calls, "wall_clock_s": run.duration_s}


@router.get("/benchmark")
def benchmark(category: str | None = None) -> dict:
    """Return approved benchmark questions with aggregated path scores."""
    candidates = [row for row in app_state.runtime.gold.list() if row.get("status") == "approved"]
    if category:
        candidates = [row for row in candidates if row.get("category") == category]
    runs = app_state.runtime.list()
    question_rows = []
    for candidate in candidates:
        path_rows = []
        for path in ("baseline", "baseline-enhanced", "agentic"):
            matching = [
                run
                for run in runs
                if run.question == candidate.get("question") and run.path == path
            ]
            if candidate.get("snapshot") is not None:
                matching = [
                    run
                    for run in matching
                    if run.snapshot
                    and run.snapshot.model_dump(by_alias=True) == candidate["snapshot"]
                ]
            evaluations = []
            for run in matching:
                if run.status != "answered":
                    continue
                try:
                    evaluations.append(
                        evaluate_run(
                            run, candidate.get("gold_triples", []), candidate.get("gold_claims")
                        )
                    )
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
            costs = [_run_cost(run) for run in matching]
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
                    "model_calls": mean([item["model_calls"] for item in costs]),
                    "tool_calls": mean([item["tool_calls"] for item in costs]),
                    "wall_clock_s": mean([item["wall_clock_s"] for item in costs]),
                }
            )
        question_rows.append(
            {
                "id": candidate.get("id"),
                "question": candidate.get("question"),
                "category": candidate.get("category", "uncategorized"),
                "theme": candidate.get("theme"),
                "difficulty": candidate.get("difficulty"),
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
    }


@router.get("/benchmark/validate")
def validate_benchmark() -> dict:
    """Validate approved gold coverage and required version metadata."""
    return app_state.runtime.gold.validate_benchmark(minimum_per_category=3)


@router.post("/benchmark/execute", status_code=202)
def execute_benchmark(request: BenchmarkExecuteRequest) -> dict[str, object]:
    """Launch explicit evaluation repetitions; ordinary runs remain single-pass."""
    if request.path not in {"baseline", "baseline-enhanced", "agentic"}:
        raise HTTPException(422, "unknown research path")
    if request.budget not in {"fast", "slow"}:
        raise HTTPException(422, "budget must be fast or slow")
    if not 1 <= request.repetitions <= 20:
        raise HTTPException(422, "repetitions must be between 1 and 20")
    config = settings()
    snapshot = request.snapshot or EvaluationSnapshot(
        corpusSnapshotId=config.corpus_snapshot_id or config.corpus_fingerprint,
        ontologyVersion=config.ontology_version,
        claimSchemaVersion="v3",
        retrievalIndexVersion=config.retrieval_index_version,
    )
    ids = [
        app_state.runtime.start(
            RunRequest(
                question=request.question,
                path=request.path,
                budget=request.budget,
                evaluation_mode=True,
                snapshot=snapshot,
                concept=request.concept if request.path == "agentic" else None,
            )
        )
        for _ in range(request.repetitions)
    ]
    return {
        "ids": ids,
        "repetitions": request.repetitions,
        "evaluation_mode": True,
        "snapshot": snapshot.model_dump(by_alias=True),
    }
