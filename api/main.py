"""Research Harness HTTP API with live baseline runs and fixture fallbacks."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from harness_baseline import Baseline20
from harness_corpus.store import CorpusStore
from harness_extraction import ExtractionEngine, OpenAICompatibleModel
from harness_observability import hello_span
from harness_observability import init as init_observability
from harness_runs import ArtifactStore, GoldCandidateStore
from harness_schema.models import Run, TraceRow
from harness_semantic import hand_written_subschema
from pydantic import BaseModel, ConfigDict
from sse_starlette import EventSourceResponse, JSONServerSentEvent

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
FIXTURES = Path(os.getenv("FIXTURES_DIR", ROOT / "web" / "fixtures"))


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str
    path: str
    budget: str


class RatingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rating: str


class CandidateUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ambiguous: bool | None = None
    readings: list[dict] | None = None
    gold_triples: list[dict] | None = None
    paper_ids: list[str] | None = None
    notes: str | None = None
    status: str | None = None


def fixture_runs() -> list[Run]:
    return [Run.model_validate_json(path.read_text()) for path in sorted(FIXTURES.glob("r-*.json"))]


class RunRuntime:
    def __init__(
        self,
        artifact_root: Path,
        datasets_root: Path,
        runner_factory: Callable[[Callable[[TraceRow], None]], Baseline20] | None = None,
    ) -> None:
        self.artifacts = ArtifactStore(artifact_root)
        self.gold = GoldCandidateStore(datasets_root / "gold" / "candidates.jsonl")
        self.custom_runner = runner_factory is not None
        self.runner_factory = runner_factory or self._runner
        self.events: dict[str, list[TraceRow]] = {}
        self.running: set[str] = set()
        self.fixture_overrides: dict[str, Run] = {}
        self.lock = threading.RLock()

    def _runner(self, sink: Callable[[TraceRow], None], enhanced: bool = False) -> Baseline20:
        schema = hand_written_subschema()
        budget_timeout = float(os.getenv("BUDGET_BASELINE_SLOW_SECONDS", "15"))
        model = OpenAICompatibleModel(timeout_s=budget_timeout)
        return Baseline20(
            CorpusStore(Path(os.getenv("CORPUS_ROOT", ROOT / "corpus_data"))),
            ExtractionEngine(model, schema),
            retrieve_k=int(os.getenv("BASELINE_RETRIEVE_K", "20")),
            enhanced=enhanced,
            edc=True,
            trace_sink=sink,
        )

    def start(self, request: RunRequest) -> str:
        if request.path not in {"baseline", "baseline-enhanced"}:
            raise ValueError("the agentic path arrives in M5")
        if request.budget not in {"fast", "slow"}:
            raise ValueError("budget must be fast or slow")
        if not request.question.strip():
            raise ValueError("question is required")
        run_id = f"r-{uuid.uuid4().hex[:8]}"
        with self.lock:
            self.events[run_id] = []
            self.running.add(run_id)

        def sink(row: TraceRow) -> None:
            with self.lock:
                self.events[run_id].append(row)

        def execute() -> None:
            try:
                runner = self.runner_factory(sink) if self.custom_runner else self._runner(
                    sink, request.path == "baseline-enhanced"
                )
                result = runner.run(
                    request.question.strip(), request.budget, run_id
                )
                self.artifacts.finalize(result["run"], result["artifacts"])
            finally:
                with self.lock:
                    self.running.discard(run_id)

        threading.Thread(target=execute, name=f"baseline-{run_id}", daemon=True).start()
        return run_id

    def list(self) -> list[Run]:
        live = self.artifacts.list_runs()
        live_ids = {run.id for run in live}
        fixtures = [self.fixture_overrides.get(run.id, run) for run in fixture_runs()]
        return live + [run for run in fixtures if run.id not in live_ids]

    def get(self, run_id: str) -> Run:
        try:
            return self.artifacts.load_run(run_id)
        except (FileNotFoundError, ValueError):
            pass
        if run_id in self.fixture_overrides:
            return self.fixture_overrides[run_id]
        for run in fixture_runs():
            if run.id == run_id:
                return run
        raise KeyError(run_id)

    def rate(self, run_id: str, value: str) -> Run:
        if value not in {"useful", "partial", "not_useful"}:
            raise ValueError("unknown rating")
        run = self.get(run_id).model_copy(update={"rating": value})
        try:
            self.artifacts.load_run(run_id)
            self.artifacts.update_run(run)
        except FileNotFoundError:
            self.fixture_overrides[run_id] = run
        return run

    def promote(self, run_id: str) -> dict:
        return self.gold.promote(self.get(run_id))

    async def stream(self, run_id: str):
        try:
            completed = self.get(run_id)
        except KeyError:
            completed = None
        if completed is not None:
            for index, row in enumerate(completed.trace):
                yield JSONServerSentEvent(
                    data=row.model_dump(by_alias=True), event="trace", id=str(index)
                )
            yield JSONServerSentEvent(data={"run_id": run_id}, event="done")
            return
        with self.lock:
            if run_id not in self.running:
                raise KeyError(run_id)
        sent = 0
        while True:
            with self.lock:
                rows = self.events.get(run_id, [])[sent:]
                active = run_id in self.running
            for row in rows:
                yield JSONServerSentEvent(
                    data=row.model_dump(by_alias=True), event="trace", id=str(sent)
                )
                sent += 1
            if not active:
                yield JSONServerSentEvent(data={"run_id": run_id}, event="done")
                return
            await asyncio.sleep(0.05)


runtime = RunRuntime(
    Path(os.getenv("ARTIFACT_ROOT", ROOT / "artifacts")),
    Path(os.getenv("DATASETS_ROOT", ROOT / "datasets")),
)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    if init_observability():
        hello_span()
    yield


app = FastAPI(title="Research Harness", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("API_CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/runs", status_code=202)
def start_run(request: RunRequest) -> dict[str, str]:
    try:
        return {"id": runtime.start(request)}
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/runs", response_model=list[Run], response_model_by_alias=True)
def list_runs() -> list[Run]:
    return runtime.list()


@app.get("/runs/{run_id}", response_model=Run, response_model_by_alias=True)
def get_run(run_id: str) -> Run:
    try:
        return runtime.get(run_id)
    except KeyError as exc:
        raise HTTPException(404, "run not found") from exc


@app.get("/runs/{run_id}/events")
def run_events(run_id: str):
    with runtime.lock:
        known = run_id in runtime.running
    try:
        runtime.get(run_id)
        known = True
    except KeyError:
        pass
    if not known:
        raise HTTPException(404, "run not found")
    return EventSourceResponse(
        runtime.stream(run_id), ping=int(os.getenv("SSE_KEEPALIVE_SECONDS", "15"))
    )


@app.post("/runs/{run_id}/rating", response_model=Run, response_model_by_alias=True)
def rate_run(run_id: str, request: RatingRequest) -> Run:
    try:
        return runtime.rate(run_id, request.rating)
    except KeyError as exc:
        raise HTTPException(404, "run not found") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/runs/{run_id}/promote", status_code=201)
def promote_run(run_id: str) -> dict:
    try:
        return runtime.promote(run_id)
    except KeyError as exc:
        raise HTTPException(404, "run not found") from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/datasets/candidates")
def list_candidates() -> list[dict]:
    return runtime.gold.list()


@app.put("/datasets/candidates/{candidate_id}")
def update_candidate(candidate_id: str, request: CandidateUpdate) -> dict:
    changes = request.model_dump(exclude_none=True)
    try:
        return runtime.gold.update(candidate_id, changes)
    except KeyError as exc:
        raise HTTPException(404, "candidate not found") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/runs/{run_id}/artifacts/{name}")
def get_artifact(run_id: str, name: str) -> dict[str, str]:
    run = get_run(run_id)
    if name not in run.artifacts:
        raise HTTPException(404, "artifact not found")
    try:
        content = runtime.artifacts.read(run_id, name).decode()
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


@app.get("/corpus/papers")
def papers() -> list[dict]:
    store = CorpusStore(Path(os.getenv("CORPUS_ROOT", ROOT / "corpus_data")))
    return store.papers


@app.get("/settings")
def settings() -> dict[str, str]:
    return {"baseline": "fast", "agentic": "slow"}
