"""Research Harness HTTP API with live baseline runs and fixture fallbacks."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from time import monotonic

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from harness_agent import AgenticResearch
from harness_baseline import Baseline20
from harness_baseline.planner import QueryPlanner
from harness_corpus.store import CorpusStore
from harness_evaluation import evaluate_run
from harness_extraction import ExtractionEngine, OpenAICompatibleModel, call_budget, model_events
from harness_observability import hello_span, run_observation
from harness_observability import init as init_observability
from harness_runs import ArtifactStore, ClaimStore, GoldCandidateStore, RelationRegistry
from harness_schema.models import (
    ConceptChoice,
    EvaluationSnapshot,
    Run,
    TraceRow,
    required_artifacts,
)
from harness_semantic import hand_written_subschema
from pydantic import BaseModel, ConfigDict
from sse_starlette import EventSourceResponse, JSONServerSentEvent

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
FIXTURES = Path(os.getenv("FIXTURES_DIR", ROOT / "web" / "fixtures"))
HEAVY_MODEL_SLOT = threading.BoundedSemaphore(
    int(os.getenv("HARNESS_MAX_CONCURRENT_HEAVY", "1"))
)
MODEL_SLOT = threading.BoundedSemaphore(
    int(os.getenv("HARNESS_MAX_CONCURRENT_RUNS", "2"))
)
LOGGER = logging.getLogger(__name__)


class RunCancelled(BaseException):
    """Cooperative cancellation raised from a run's trace sink.

    It derives from BaseException so the research pipeline's broad
    ``except Exception`` handlers cannot swallow it and keep working.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def run_deadline_seconds(path: str, budget: str) -> float:
    """Hard wall-clock limit for one run, independent of per-call timeouts."""
    default = {"fast": 300.0, "slow": 900.0}[budget]
    return float(os.getenv(f"HARNESS_RUN_DEADLINE_{budget.upper()}_S", default))


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str
    path: str
    budget: str
    evaluation_mode: bool = False
    snapshot: EvaluationSnapshot | None = None
    # Agent-proposed concept chosen by the user; required before agentic research.
    concept: ConceptChoice | None = None
    parent_run_id: str | None = None


class RatingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rating: str


class CandidateUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ambiguous: bool | None = None
    readings: list[dict] | None = None
    gold_triples: list[dict] | None = None
    gold_claims: list[dict] | None = None
    paper_ids: list[str] | None = None
    notes: str | None = None
    status: str | None = None
    category: str | None = None
    gold_version: str | None = None
    corpus_fingerprint: str | None = None
    ontology_hash: str | None = None
    reviewer: str | None = None
    reviewed_at: str | None = None
    accepted_readings: list[dict] | None = None
    evidence_refs: list[str] | None = None


class ReadingPick(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pick: int


class EvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gold_triples: list[dict]
    gold_claims: list[dict] | None = None


class BenchmarkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repetitions: int = 1
    category: str | None = None


class BenchmarkExecuteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str
    path: str
    budget: str = "slow"
    repetitions: int = 3
    snapshot: EvaluationSnapshot | None = None
    # Agentic only: the concept to research. When omitted, each repetition lets
    # the agent propose concepts and researches its top-ranked grounded one.
    concept: ConceptChoice | None = None


class SettingsUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    baseline: str | None = None
    agentic: str | None = None


class RelationReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str


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
        self.claims = ClaimStore(artifact_root / "claim_graph.sqlite")
        self.relations = RelationRegistry(artifact_root / "relation_registry.sqlite")
        self.gold = GoldCandidateStore(datasets_root / "gold" / "candidates.jsonl")
        self.custom_runner = runner_factory is not None
        self.runner_factory = runner_factory or self._runner
        self.events: dict[str, list[TraceRow]] = {}
        self.running: set[str] = set()
        self.stop_requested: set[str] = set()
        self.settings = {"baseline": "fast", "agentic": "slow"}
        self.fixture_overrides: dict[str, Run] = {}
        self.picks: dict[str, tuple[str, int, ConceptChoice]] = {}
        self.lock = threading.RLock()
        self._recover_interrupted()

    def _runner(
        self,
        sink: Callable[[TraceRow], None],
        enhanced: bool = False,
        model_name: str | None = None,
        timeout_s: float | None = None,
        refinement_model: OpenAICompatibleModel | None = None,
    ) -> Baseline20:
        schema = hand_written_subschema()
        # Qwen is shared by every path for claim construction. Its structured
        # responses routinely exceed the old 15-second transport deadline;
        # that limit turned successful research into provider failures.
        budget_timeout = float(os.getenv("BUDGET_BASELINE_SLOW_SECONDS", "90"))
        model = OpenAICompatibleModel(model=model_name, timeout_s=timeout_s or budget_timeout)
        return Baseline20(
            CorpusStore(Path(os.getenv("CORPUS_ROOT", ROOT / "corpus_data"))),
            ExtractionEngine(model, schema),
            planner=QueryPlanner(model),
            retrieve_k=int(os.getenv("BASELINE_RETRIEVE_K", "20")),
            enhanced=enhanced,
            edc=True,
            trace_sink=sink,
            refine=enhanced,
            refinement_model=refinement_model,
            claim_lookup=self.claims.search,
            relation_proposer=lambda name, definition, run_id: self.relations.propose(
                name, definition, run_id=run_id, domain="Paper", range_="Resource"
            ),
        )

    def start(self, request: RunRequest) -> str:
        if request.path not in {"agentic", "baseline", "baseline-enhanced"}:
            raise ValueError("unknown research path")
        if request.budget not in {"fast", "slow"}:
            raise ValueError("budget must be fast or slow")
        if not request.question.strip():
            raise ValueError("question is required")
        run_id = f"r-{uuid.uuid4().hex[:8]}"
        with self.lock:
            self.events[run_id] = []
            self.running.add(run_id)
        self._mark_pending(run_id, request)
        run_started = monotonic()
        deadline = run_started + run_deadline_seconds(request.path, request.budget)
        question = request.question.strip()

        def check_cancelled() -> None:
            with self.lock:
                stopped = run_id in self.stop_requested
            if stopped:
                raise RunCancelled("stopped")
            if monotonic() > deadline:
                raise RunCancelled("deadline_exceeded")

        def sink(row: TraceRow) -> None:
            # One clock for the whole run: pipeline, agent and model-call rows
            # are stamped with seconds since the run started.
            row = row.model_copy(update={"at_s": round(monotonic() - run_started, 2)})
            with self.lock:
                self.events[run_id].append(row)
            # Every pipeline stage reports through the sink, so this is where a
            # stop request or the run deadline takes effect.
            check_cancelled()

        def acquire(*slots: threading.BoundedSemaphore) -> list[threading.BoundedSemaphore]:
            held: list[threading.BoundedSemaphore] = []
            announced = False
            try:
                for slot in slots:
                    while not slot.acquire(timeout=0.5):
                        if not announced:
                            announced = True
                            with self.lock:
                                self.events[run_id].append(TraceRow(
                                    atS=0.0, action="Queued for model capacity",
                                    returned="waiting for an earlier run to finish",
                                ))
                        check_cancelled()
                    held.append(slot)
            except BaseException:
                for slot in held:
                    slot.release()
                raise
            return held

        def execute() -> None:
            result: dict | None = None
            try:
                with (
                    run_observation(run_id, question, request.path, request.budget) as observation,
                    call_budget(deadline, lambda: run_id in self.stop_requested),
                    model_events(lambda action, detail: sink(TraceRow(
                        atS=round(monotonic() - run_started, 2), action=action, returned=detail,
                    ))),
                ):
                    result = self._execute(request, question, run_id, sink, acquire)
                    if observation is not None:
                        observation.update(
                            output={
                                "status": result["run"].status,
                                "triple_count": sum(len(item.triples) for item in result["run"].readings),
                                "claim_count": sum(len(item.claims) for item in result["run"].readings),
                            }
                        )
                with self.lock:
                    trace = list(self.events[run_id])
                # The saved trace is exactly what streamed live, including
                # model-call start/finish rows.
                result["artifacts"]["trace.json"] = json.dumps(
                    [item.model_dump(by_alias=True) for item in trace], indent=2
                ).encode()
                run = result["run"].model_copy(update={
                    "evaluation_mode": request.evaluation_mode,
                    "snapshot": request.snapshot,
                    "parent_run_id": request.parent_run_id,
                    "trace": trace,
                })
                result["run"] = run
                with self.lock:
                    stopped = run_id in self.stop_requested
                if stopped:
                    run = run.model_copy(update={"status": "stopped", "chosen_reading": None})
                    result["run"] = run
                elif run.status == "incomplete" and monotonic() > deadline:
                    # A model call refused to start after the deadline; report
                    # the cause, not the provider symptom.
                    run = run.model_copy(update={"reason": "deadline_exceeded"})
                    result["run"] = run
                self.artifacts.finalize(run, result["artifacts"])
                if run.status == "answered" and run.chosen_reading is not None:
                    self.claims.upsert(run.readings[run.chosen_reading].claims)
            except RunCancelled as cancelled:
                stopped = cancelled.reason == "stopped"
                with self.lock:
                    self.events[run_id].append(TraceRow(
                        atS=0.0,
                        action="Run stopped" if stopped else "Run deadline exceeded",
                        returned="stopped by user" if stopped
                        else f"exceeded {run_deadline_seconds(request.path, request.budget):.0f}s",
                    ))
                self._finalize_terminal(
                    request, question, run_id,
                    status="stopped" if stopped else "incomplete",
                    reason=None if stopped else "deadline_exceeded",
                    message="Run stopped." if stopped else "Run exceeded its time limit.",
                )
            except Exception as exc:
                LOGGER.exception("Run %s failed unexpectedly", run_id)
                with self.lock:
                    self.events[run_id].append(
                        TraceRow(atS=0.0, action="Run incomplete", returned="internal_error")
                    )
                self._finalize_terminal(
                    request, question, run_id, status="incomplete", reason="internal_error",
                    message=f"Run failed unexpectedly: {type(exc).__name__}",
                )
            finally:
                if (self.artifacts.root / run_id / "run.json").exists():
                    self._pending_path(run_id).unlink(missing_ok=True)
                with self.lock:
                    self.running.discard(run_id)
                    self.stop_requested.discard(run_id)

        threading.Thread(target=execute, name=f"run-{run_id}", daemon=True).start()
        return run_id

    def _execute(self, request: RunRequest, question: str, run_id: str, sink, acquire) -> dict:
        """Run one request, holding model slots only while research runs."""
        if request.path != "agentic" or self.custom_runner:
            runner = self.runner_factory(sink) if self.custom_runner else self._runner(
                sink, request.path == "baseline-enhanced"
            )
            kwargs = {"concept": request.concept} if request.concept else {}
            held = acquire(MODEL_SLOT)
            try:
                return runner.run(question, request.budget, run_id, **kwargs)
            finally:
                for slot in held:
                    slot.release()

        # The lite model controls MCP tool use. The high-quality model handles
        # concept interpretation and evidence review; the small model remains
        # responsible for claim construction in the shared baseline pipeline.
        agent_timeout = float(os.getenv(
            "BUDGET_AGENTIC_SLOW_SECONDS" if request.budget == "slow" else "BUDGET_AGENTIC_FAST_SECONDS",
            "120" if request.budget == "slow" else "45",
        ))
        agent_model = OpenAICompatibleModel(
            model=os.getenv("HARNESS_MODEL_AGENT", "lite-model"), timeout_s=agent_timeout,
        )
        high_model = OpenAICompatibleModel(
            model=os.getenv("HARNESS_MODEL_HIGH", "heavy-model"), timeout_s=agent_timeout,
        )
        baseline = self._runner(sink, True, refinement_model=high_model)
        runner = AgenticResearch(
            baseline, sink, agent_model=agent_model, high_model=high_model,
        )
        concept = request.concept
        proposal: dict | None = None
        if concept is None:
            # One short planning call: it needs no research slot, so it never
            # waits behind another user's long-running research.
            proposal = runner.propose(question, request.budget, run_id)
            if not request.evaluation_mode or proposal["run"].status != "awaiting_reading":
                return proposal
            # Evaluation runs are unattended: research the agent's top-ranked
            # grounded concept and record that it was selected automatically.
            concept = proposal["run"].concepts[0].model_copy(update={"auto_selected": True})
            sink(TraceRow(atS=0.0, action="Concept auto-selected for evaluation",
                          returned=concept.label))
        held = acquire(HEAVY_MODEL_SLOT, MODEL_SLOT)
        try:
            result = runner.run(question, request.budget, run_id, concept=concept)
        finally:
            for slot in held:
                slot.release()
        if proposal is not None:
            result["run"] = result["run"].model_copy(update={"concepts": proposal["run"].concepts})
            result["artifacts"]["plans.json"] = proposal["artifacts"]["plans.json"]
        return result

    # ─── Crash safety: a run that was active when the API stopped ──────────

    def _pending_path(self, run_id: str) -> Path:
        return self.artifacts.root / run_id / "pending.json"

    def _mark_pending(self, run_id: str, request: RunRequest) -> None:
        try:
            path = self._pending_path(run_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(request.model_dump_json())
        except OSError:
            LOGGER.warning("Could not write pending marker for %s", run_id, exc_info=True)

    def _recover_interrupted(self) -> None:
        """Close runs that were active when the API process last stopped."""
        root = self.artifacts.root
        if not root.exists():
            return
        for marker in root.glob("*/pending.json"):
            run_id = marker.parent.name
            if (marker.parent / "run.json").exists():
                marker.unlink(missing_ok=True)
                continue
            try:
                request = RunRequest.model_validate_json(marker.read_text())
            except (OSError, ValueError):
                LOGGER.warning("Unreadable pending marker for %s", run_id)
                continue
            self._finalize_terminal(
                request, request.question.strip(), run_id, status="incomplete",
                reason="interrupted", message="The API stopped while this run was active.",
            )
            if (marker.parent / "run.json").exists():
                marker.unlink(missing_ok=True)
            LOGGER.warning("Marked interrupted run %s as incomplete", run_id)

    def _finalize_terminal(
        self, request: RunRequest, question: str, run_id: str, *,
        status: str, reason: str | None, message: str,
    ) -> None:
        with self.lock:
            trace = list(self.events.get(run_id, []))
        run = Run(
            id=run_id, startedAt=datetime.now(UTC), durationS=0.0, question=question,
            path=request.path, budget=request.budget, status=status, reason=reason,
            trace=trace, concept=request.concept, parentRunId=request.parent_run_id,
            evaluationMode=request.evaluation_mode, snapshot=request.snapshot,
            artifacts=list(required_artifacts(request.path)),
        )
        try:
            self.artifacts.finalize(run, {
                "answer.md": message.encode(),
                "trace.json": json.dumps([item.model_dump(by_alias=True) for item in trace]).encode(),
            })
        except Exception:
            LOGGER.exception("Could not persist terminal state for run %s", run_id)

    def list(self) -> list[Run]:
        live = self.artifacts.list_runs()
        if os.getenv("HARNESS_INCLUDE_FIXTURES", "true").casefold() in {"0", "false", "no"}:
            # Live deployments list only real runs; fixture ids stay readable.
            return live
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

    def pick_reading(self, run_id: str, index: int) -> Run:
        run = self.get(run_id)
        if run.path not in {"agentic", "baseline", "baseline-enhanced"}:
            raise ValueError("reading selection is unavailable for this path")
        if run.status != "awaiting_reading":
            raise ValueError("run is not waiting for a reading")
        if index < 0 or index >= len(run.readings):
            raise ValueError("reading index is out of range")
        with self.lock:
            existing = self.picks.get(run_id)
            if existing is None:
                reading = run.readings[index]
                concept = (
                    run.concepts[index] if index < len(run.concepts)
                    else ConceptChoice(label=reading.label, gloss=reading.gloss,
                                       classes=reading.classes)
                )
                # The question is kept verbatim; the concept is passed as
                # structured input so routing, matching and gold lookup all
                # see the user's actual question.
                child_id = self.start(RunRequest(
                    question=run.question, path=run.path, budget=run.budget,
                    concept=concept, parent_run_id=run.id,
                    evaluation_mode=run.evaluation_mode, snapshot=run.snapshot,
                ))
                existing = self.picks[run_id] = (child_id, index, concept)
        child_id, picked, concept = existing
        return run.model_copy(update={
            "id": child_id, "started_at": datetime.now(UTC), "duration_s": 0.0,
            "chosen_reading": picked, "concept": concept, "parent_run_id": run.id,
        })

    def stop(self, run_id: str) -> Run | None:
        with self.lock:
            if run_id in self.running:
                self.stop_requested.add(run_id)
                return None
        run = self.get(run_id)
        if run.status in {"answered", "corpus_absent", "no_evidence_found", "awaiting_reading"}:
            run = run.model_copy(update={"status": "stopped", "chosen_reading": None})
            try:
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
    # Telemetry must not delay API availability when the remote collector is
    # slow or temporarily unavailable, including during client initialization.
    asyncio.create_task(
        asyncio.to_thread(lambda: init_observability() and hello_span())
    )
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


@app.post("/runs/{run_id}/reading", response_model=Run, response_model_by_alias=True)
def pick_reading(run_id: str, request: ReadingPick) -> Run:
    try:
        return runtime.pick_reading(run_id, request.pick)
    except KeyError as exc:
        raise HTTPException(404, "run not found") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/runs/{run_id}/stop", response_model=Run | None, response_model_by_alias=True)
def stop_run(run_id: str) -> Run | None:
    try:
        return runtime.stop(run_id)
    except KeyError as exc:
        raise HTTPException(404, "run not found") from exc


@app.post("/runs/{run_id}/evaluate")
def evaluate_run_endpoint(run_id: str, request: EvaluationRequest) -> dict:
    try:
        return evaluate_run(runtime.get(run_id), request.gold_triples, request.gold_claims)
    except KeyError as exc:
        raise HTTPException(404, "run not found") from exc


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


@app.get("/schema/relations")
def list_relations() -> list[dict]:
    return runtime.relations.list()


@app.put("/schema/relations/{relation_id}")
def review_relation(relation_id: str, request: RelationReviewRequest) -> dict:
    try:
        return runtime.relations.review(relation_id, request.status)
    except KeyError as exc:
        raise HTTPException(404, "relation not found") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


def _score_aggregate(scores: list[dict], metric: str) -> dict[str, float | None]:
    values = [score[metric] for score in scores]
    return {
        "mean": mean(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


@app.get("/benchmark")
def benchmark(category: str | None = None) -> dict:
    """Return approved benchmark questions with aggregated path scores."""
    candidates = [row for row in runtime.gold.list() if row.get("status") == "approved"]
    if category:
        candidates = [row for row in candidates if row.get("category") == category]
    runs = runtime.list()
    question_rows = []
    for candidate in candidates:
        path_rows = []
        for path in ("baseline", "baseline-enhanced", "agentic"):
            matching = [run for run in runs if run.question == candidate.get("question") and run.path == path]
            if candidate.get("snapshot") is not None:
                matching = [run for run in matching if run.snapshot and
                            run.snapshot.model_dump(by_alias=True) == candidate["snapshot"]]
            evaluations = []
            for run in matching:
                if run.status != "answered":
                    continue
                try:
                    evaluations.append(evaluate_run(
                        run, candidate.get("gold_triples", []), candidate.get("gold_claims")
                    ))
                except (ValueError, TypeError, KeyError) as exc:
                    LOGGER.warning("Cannot score run %s against %s: %s", run.id, candidate.get("id"), exc)
            eligible = [result for result in evaluations if result["eligible_for_path_score"]]
            scores = [result["chosen"] for result in eligible if result["chosen"] is not None]
            claim_scores = [
                reading["claim_score"]["claim"]
                for result in eligible for reading in result["readings"]
                if reading["index"] == result["chosen_reading"] and reading["claim_score"] is not None
            ]
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
                }
            )
        question_rows.append(
            {
                "id": candidate.get("id"),
                "question": candidate.get("question"),
                "category": candidate.get("category", "uncategorized"),
                "gold_version": candidate.get("gold_version"),
                "corpus_fingerprint": candidate.get("corpus_fingerprint"),
                "ontology_hash": candidate.get("ontology_hash"),
                "snapshot": candidate.get("snapshot"),
                "paths": path_rows,
            }
        )
    return {"questions": question_rows, "categories": sorted({row["category"] for row in question_rows})}


@app.get("/benchmark/validate")
def validate_benchmark() -> dict:
    """Validate approved gold coverage and required version metadata."""
    return runtime.gold.validate_benchmark(minimum_per_category=3)


@app.post("/benchmark/execute", status_code=202)
def execute_benchmark(request: BenchmarkExecuteRequest) -> dict[str, object]:
    """Launch explicit evaluation repetitions; ordinary runs remain single-pass."""
    if request.path not in {"baseline", "baseline-enhanced", "agentic"}:
        raise HTTPException(422, "unknown research path")
    if request.budget not in {"fast", "slow"}:
        raise HTTPException(422, "budget must be fast or slow")
    if not 1 <= request.repetitions <= 20:
        raise HTTPException(422, "repetitions must be between 1 and 20")
    snapshot = request.snapshot or EvaluationSnapshot(
        corpusSnapshotId=os.getenv("CORPUS_SNAPSHOT_ID", os.getenv("CORPUS_FINGERPRINT", "unknown")),
        ontologyVersion=os.getenv("ONTOLOGY_VERSION", "unknown"),
        claimSchemaVersion="v3",
        retrievalIndexVersion=os.getenv("RETRIEVAL_INDEX_VERSION", "unknown"),
    )
    ids = [
        runtime.start(RunRequest(question=request.question, path=request.path, budget=request.budget,
                                 evaluation_mode=True, snapshot=snapshot,
                                 concept=request.concept if request.path == "agentic" else None))
        for _ in range(request.repetitions)
    ]
    return {"ids": ids, "repetitions": request.repetitions, "evaluation_mode": True,
            "snapshot": snapshot.model_dump(by_alias=True)}


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


def _question_sets() -> list[Path]:
    root = Path(os.getenv("DATASETS_ROOT", ROOT / "datasets"))
    configured = os.getenv("HARNESS_QUESTION_SETS", "question_matrix_20.json")
    paths = []
    for name in configured.split(","):
        name = name.strip()
        if name:
            path = Path(name)
            paths.append(path if path.is_absolute() else root / path)
    return paths


@app.get("/questions")
def questions() -> dict[str, list[dict]]:
    """Questions the UI offers for one-click runs: curated sets, gold rows, recent runs."""
    items: list[dict] = []
    seen: set[str] = set()

    def add(question: object, **meta: object) -> None:
        text = " ".join(str(question or "").split())
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            items.append({"question": text, **meta})

    for path in _question_sets():
        try:
            rows = json.loads(path.read_text())
        except (OSError, ValueError):
            LOGGER.warning("Question set %s is unreadable", path)
            continue
        for row in rows if isinstance(rows, list) else rows.get("questions", []):
            if isinstance(row, dict):
                add(row.get("question"), id=row.get("id"), category=row.get("category", "curated"),
                    type=row.get("type"), expected=row.get("expected"), source=path.stem)
    for row in runtime.gold.list():
        add(row.get("question"), id=row.get("id"), category="gold", type=row.get("status"),
            source="gold")
    for run in runtime.artifacts.list_runs()[:50]:
        if run.path in {"agentic", "baseline", "baseline-enhanced"} and not run.evaluation_mode:
            add(run.question, id=run.id, category="recent", type=run.status, source="runs")
    recent = [item for item in items if item["category"] == "recent"][:10]
    return {"questions": [item for item in items if item["category"] != "recent"] + recent}


@app.get("/settings")
def settings() -> dict[str, str]:
    return dict(runtime.settings)


@app.put("/settings")
def update_settings(request: SettingsUpdate) -> dict[str, str]:
    changes = request.model_dump(exclude_none=True)
    if any(value not in {"fast", "slow"} for value in changes.values()):
        raise HTTPException(422, "budgets must be fast or slow")
    runtime.settings.update(changes)
    return dict(runtime.settings)
