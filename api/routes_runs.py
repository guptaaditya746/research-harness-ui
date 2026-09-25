"""Run endpoints: start, list, read, stream, rate, pick a reading, stop, evaluate, artifacts."""

from __future__ import annotations

import json

import app_state
from api_models import EvaluationRequest, RatingRequest, ReadingPick, RunRequest
from capacity import consume_public_reservation
from fastapi import APIRouter, Header, HTTPException
from harness_config import settings
from harness_evaluation import evaluate_run
from harness_schema.models import Run
from sse_starlette import EventSourceResponse

router = APIRouter()


@router.post("/runs", status_code=202)
def start_run(
    request: RunRequest,
    x_public_reservation: str | None = Header(default=None),
) -> dict[str, str]:
    reserved_slots = None
    if request.origin == "public":
        reserved_slots = consume_public_reservation(x_public_reservation)
        if reserved_slots is None:
            raise HTTPException(503, "service busy")
    try:
        return {"id": app_state.runtime.start(request, reserved_slots=reserved_slots)}
    except Exception as exc:
        if reserved_slots:
            for slot in reserved_slots:
                slot.release()
        if isinstance(exc, ValueError):
            raise HTTPException(422, str(exc)) from exc
        raise


@router.get("/runs", response_model=list[Run], response_model_by_alias=True)
def list_runs() -> list[Run]:
    return app_state.runtime.list()


@router.get("/runs/{run_id}", response_model=Run, response_model_by_alias=True)
def get_run(run_id: str) -> Run:
    try:
        return app_state.runtime.get(run_id)
    except KeyError as exc:
        raise HTTPException(404, "run not found") from exc


@router.get("/runs/{run_id}/events")
def run_events(run_id: str):
    with app_state.runtime.lock:
        known = run_id in app_state.runtime.running
    try:
        app_state.runtime.get(run_id)
        known = True
    except KeyError:
        pass
    if not known:
        raise HTTPException(404, "run not found")
    return EventSourceResponse(app_state.runtime.stream(run_id), ping=settings().sse_keepalive_seconds)


@router.post("/runs/{run_id}/rating", response_model=Run, response_model_by_alias=True)
def rate_run(run_id: str, request: RatingRequest) -> Run:
    try:
        return app_state.runtime.rate(run_id, request.rating)
    except KeyError as exc:
        raise HTTPException(404, "run not found") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/runs/{run_id}/reading", response_model=Run, response_model_by_alias=True)
def pick_reading(run_id: str, request: ReadingPick) -> Run:
    try:
        return app_state.runtime.pick_reading(run_id, request.pick)
    except KeyError as exc:
        raise HTTPException(404, "run not found") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/runs/{run_id}/stop", response_model=Run | None, response_model_by_alias=True)
def stop_run(run_id: str) -> Run | None:
    try:
        return app_state.runtime.stop(run_id)
    except KeyError as exc:
        raise HTTPException(404, "run not found") from exc


@router.post("/runs/{run_id}/evaluate")
def evaluate_run_endpoint(run_id: str, request: EvaluationRequest) -> dict:
    try:
        return evaluate_run(app_state.runtime.get(run_id), request.gold_triples, request.gold_claims)
    except KeyError as exc:
        raise HTTPException(404, "run not found") from exc


@router.post("/runs/{run_id}/promote", status_code=201)
def promote_run(run_id: str) -> dict:
    try:
        return app_state.runtime.promote(run_id)
    except KeyError as exc:
        raise HTTPException(404, "run not found") from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc



@router.get("/runs/{run_id}/artifacts/{name}")
def get_artifact(run_id: str, name: str) -> dict[str, str]:
    run = get_run(run_id)
    if name not in run.artifacts:
        raise HTTPException(404, "artifact not found")
    try:
        content = app_state.runtime.artifacts.read(run_id, name).decode()
        return {"name": name, "content": content}
    except FileNotFoundError:
        pass
    if name == "run.json":
        content = run.model_dump_json(by_alias=True, indent=2)
    elif name == "answer.md":
        content = run.readings[run.chosen_reading].answer if run.chosen_reading is not None else ""
    elif name == "query.rq":
        content = run.readings[run.chosen_reading].sparql if run.chosen_reading is not None else ""
    elif name == "plans.json":
        content = json.dumps(run.plans, indent=2)
    elif name == "trace.json":
        content = json.dumps([row.model_dump(by_alias=True) for row in run.trace], indent=2)
    else:
        content = "Fixture placeholder"
    return {"name": name, "content": content}
