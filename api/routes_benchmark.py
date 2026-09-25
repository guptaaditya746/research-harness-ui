"""Benchmark endpoints: scores per path against approved gold, and evaluation repetitions."""

from __future__ import annotations

import app_state
from api_models import BenchmarkExecuteRequest, RunRequest
from fastapi import APIRouter, HTTPException
from harness_config import settings
from harness_evaluation import load_relations
from harness_evaluation.benchmark import build_benchmark
from harness_evaluation.judge import CachedJudge
from harness_schema.models import EvaluationSnapshot, Run

router = APIRouter()


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
def benchmark(category: str | None = None, judge: bool = True) -> dict:
    """Approved benchmark questions with aggregated path scores (see harness_evaluation.benchmark).

    ``judge=false`` reports the soft level from reviewer variants only (use it when the judge's
    agreement with people is below the bar, see labelling docs/EVALUATION.md).
    """
    candidates = [row for row in app_state.runtime.gold.list() if row.get("status") == "approved"]
    if category:
        candidates = [row for row in candidates if row.get("category") == category]
    # Cache-only judge: opening the page never starts model calls (`make evaluate ARGS=--judge` fills it).
    result = build_benchmark(candidates, app_state.runtime.list(), relations=load_relations(),
                             judge=CachedJudge() if judge else None, cost=_run_cost)
    result["judge"] = "cache" if judge else "off"
    return result


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
