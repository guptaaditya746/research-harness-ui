"""M0 API: generated fixtures and artifact previews only."""

import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from harness_observability import hello_span
from harness_observability import init as init_observability
from harness_schema.models import Run

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(os.getenv("FIXTURES_DIR", ROOT / "web" / "fixtures"))


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    if init_observability():
        hello_span()
    yield


app = FastAPI(title="Research Harness M0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("API_CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_methods=["GET"],
    allow_headers=["*"],
)


def fixture_runs() -> list[Run]:
    return [Run.model_validate_json(path.read_text()) for path in sorted(FIXTURES.glob("r-*.json"))]


@app.get("/runs", response_model=list[Run], response_model_by_alias=True)
def list_runs() -> list[Run]:
    return fixture_runs()


@app.get("/runs/{run_id}", response_model=Run, response_model_by_alias=True)
def get_run(run_id: str) -> Run:
    for run in fixture_runs():
        if run.id == run_id:
            return run
    raise HTTPException(404, "run not found")


@app.get("/runs/{run_id}/artifacts/{name}")
def get_artifact(run_id: str, name: str) -> dict[str, str]:
    run = get_run(run_id)
    if name not in run.artifacts:
        raise HTTPException(404, "artifact not found")
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
        content = "Fixture placeholder: this artifact is created by the live pipeline in a later milestone."
    return {"name": name, "content": content}


@app.get("/corpus/papers")
def papers() -> list[dict[str, str | int]]:
    return [
        {
            "title": "Break It Down: A Question Understanding Benchmark",
            "year": 2020,
            "status": "Indexed",
        },
        {
            "title": "AmbigQA: Answering Ambiguous Open-domain Questions",
            "year": 2020,
            "status": "Indexed",
        },
        {
            "title": "OneKE: A Dockerized Schema-Guided Extraction System",
            "year": 2024,
            "status": "Indexed",
        },
        {"title": "Dense Passage Retrieval for Open-Domain QA", "year": 2020, "status": "Indexed"},
        {
            "title": "Measuring Ambiguity in Open-ended Research Questions",
            "year": 2023,
            "status": "Parsing",
        },
    ]


@app.get("/settings")
def settings() -> dict[str, str]:
    return {"baseline": "fast", "agentic": "slow"}
