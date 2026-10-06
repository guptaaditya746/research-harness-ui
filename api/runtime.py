"""Run lifecycle: start, stream, stop, rate and pick readings; one thread per run.

``RunRuntime`` builds the research runner for a request (Fast, Adaptive or
Agentic), holds model slots only while research runs, stamps every trace row
on one run clock, enforces Stop and the run deadline through the trace sink,
and persists results and terminal states (including runs interrupted by an
API restart).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic

from api_models import RunRequest
from capacity import run_slots
from harness_agent import AgenticResearch
from harness_baseline import Baseline20, ExternalHarness
from harness_baseline import evidence_doc as evidence_docs
from harness_baseline.planner import QueryPlanner, corpus_profile
from harness_config import settings, ui
from harness_corpus.store import CorpusStore
from harness_extraction import (
    ExtractionEngine,
    ModelFailure,
    OpenAICompatibleModel,
    call_budget,
    model_events,
)
from harness_observability import run_observation
from harness_runs import ArtifactStore, ClaimStore, GoldCandidateStore, RelationRegistry
from harness_schema.models import ConceptChoice, Run, TraceRow, required_artifacts
from harness_semantic import hand_written_subschema
from paths import corpus_root, evidence_doc_root, fixtures_dir
from sse_starlette import JSONServerSentEvent

LOGGER = logging.getLogger(__name__)
_PROMPT_REF = re.compile(r"prompt ([\w/]+)@(v\d+):([0-9a-f]{8})")


def prompt_versions(trace: list[TraceRow]) -> dict[str, str]:
    """Prompt templates used in a run, from its model-call rows: name -> "set:hash"."""
    found: dict[str, str] = {}
    for row in trace:
        for name, prompt_set, digest in _PROMPT_REF.findall(row.returned or ""):
            found[name] = f"{prompt_set}:{digest}"
    return found


class RunCancelled(BaseException):
    """Cooperative cancellation raised from a run's trace sink.

    It derives from BaseException so the research pipeline's broad
    ``except Exception`` handlers cannot swallow it and keep working.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def external_configured() -> bool:
    config = settings()
    return bool(config.external_command.strip() and config.external_config.strip())


class _NoHarnessModel:
    """Stands in for a model on the External path, where only the external agent calls models."""

    model = "none"

    def complete_json(self, system: str, user: str) -> dict:
        raise ModelFailure("the External path makes no harness model calls")


def run_deadline_seconds(path: str, budget: str) -> float:
    """Hard wall-clock limit for one run, independent of per-call timeouts."""
    return settings().run_deadline_seconds(path, budget)


def fixture_runs() -> list[Run]:
    return [Run.model_validate_json(path.read_text()) for path in sorted(fixtures_dir().glob("r-*.json"))]


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
        # Runs waiting for a model slot (a subset of ``running``); the model status endpoint reports them.
        self.queued: set[str] = set()
        self.stop_requested: set[str] = set()
        self.settings = {"baseline": "fast", "agentic": "slow"}
        # Settings changed on the Settings page (harness_config.ui), kept in a file and applied again here.
        self.ui_file = settings().ui_settings_file or artifact_root / "ui_settings.json"
        self.ui_originals: dict[str, str | None] = {}
        self.ui_overrides = ui.load_file(self.ui_file, self.ui_originals)
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
        schema_model: OpenAICompatibleModel | None = None,
        max_passages: int | None = None,
    ) -> Baseline20:
        schema = hand_written_subschema()
        # Qwen is shared by every path for claim construction. Its structured
        # responses routinely exceed the old 15-second transport deadline;
        # that limit turned successful research into provider failures.
        config = settings()
        budget_timeout = config.budget_baseline_slow_seconds
        model = OpenAICompatibleModel(model=model_name, timeout_s=timeout_s or budget_timeout)
        edc_model = OpenAICompatibleModel(
            model=config.model_edc or model_name or config.model_small,
            timeout_s=timeout_s or budget_timeout,
        )
        corpus = CorpusStore(corpus_root())
        return Baseline20(
            corpus,
            ExtractionEngine(model, schema, schema_model=schema_model, edc_model=edc_model),
            planner=QueryPlanner(model, profile=corpus_profile(corpus)),
            retrieve_k=config.baseline_retrieve_k,
            max_passages=(
                max_passages
                if max_passages is not None
                else config.adaptive_max_passages
                if enhanced
                else None
            ),
            enhanced=enhanced,
            edc=True,
            trace_sink=sink,
            refine=enhanced,
            refinement_model=refinement_model,
            # Claims accepted in earlier runs feed retrieval only when switched on: on by default they
            # carry earlier runs of the same question into later ones (evaluation leakage).
            claim_lookup=self.claims.search if settings().claim_reuse else None,
            relation_proposer=lambda name, definition, run_id, domain, range_: (
                self.relations.propose(
                    name, definition, run_id=run_id, domain=domain, range_=range_
                )
            ),
        )

    def _external_runner(self, sink: Callable[[TraceRow], None]) -> ExternalHarness:
        """The External path: no harness model is called (claim enrichment finds none and skips)."""
        config = settings()
        return ExternalHarness(
            CorpusStore(corpus_root()), ExtractionEngine(_NoHarnessModel(), hand_written_subschema()), trace_sink=sink,
            command=config.external_command, config_path=config.external_config,
        )

    def start(
        self,
        request: RunRequest,
        reserved_slots: list[threading.BoundedSemaphore] | None = None,
    ) -> str:
        if request.path not in {"agentic", "baseline", "baseline-enhanced", "external"}:
            raise ValueError("unknown research path")
        if request.path == "external" and not external_configured():
            raise ValueError("the External path is off: set HARNESS_EXTERNAL_COMMAND and HARNESS_EXTERNAL_CONFIG "
                             "(see the external-harness README)")
        if request.origin not in {"private", "public"}:
            raise ValueError("unknown run origin")
        if request.origin == "public" and reserved_slots is None:
            raise ValueError("public model capacity reservation is required")
        if request.deadline_seconds is not None and not 1 <= request.deadline_seconds <= 900:
            raise ValueError("invalid deadline")
        if request.budget not in {"fast", "slow"}:
            raise ValueError("budget must be fast or slow")
        if not request.question.strip():
            raise ValueError("question is required")
        run_input = request.run_input or settings().run_input
        if run_input not in {"corpus", "evidence_doc"}:
            raise ValueError("run_input must be corpus or evidence_doc")
        request = request.model_copy(update={"run_input": run_input})
        recorded_settings = ui.values()
        doc = None
        if run_input == "evidence_doc":
            # Fail before the run starts: without D there is nothing to read.
            doc = evidence_docs.find(evidence_doc_root(), request.question.strip())
            if doc is None:
                raise ValueError(f"no evidence document for this question in {evidence_doc_root()} (build it "
                                 "with `labelling evidence-docs`, or set HARNESS_EVIDENCE_DOC_ROOT)")
        run_id = f"r-{uuid.uuid4().hex[:8]}"
        with self.lock:
            self.events[run_id] = []
            self.running.add(run_id)
        self._mark_pending(run_id, request)
        run_started = monotonic()
        configured_deadline = run_deadline_seconds(request.path, request.budget)
        requested_deadline = request.deadline_seconds
        deadline = run_started + (
            min(configured_deadline, requested_deadline)
            if requested_deadline
            else configured_deadline
        )
        question = request.question.strip()
        held_reservation = reserved_slots

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
            nonlocal held_reservation
            if request.origin == "public" and held_reservation is not None:
                held, held_reservation = held_reservation, None
                return held
            held: list[threading.BoundedSemaphore] = []
            announced = False
            try:
                for slot in slots:
                    while not slot.acquire(timeout=0.5):
                        if not announced:
                            announced = True
                            with self.lock:
                                self.queued.add(run_id)
                                self.events[run_id].append(
                                    TraceRow(
                                        atS=0.0,
                                        action="Queued for model capacity",
                                        returned="waiting for an earlier run to finish",
                                    )
                                )
                        check_cancelled()
                    held.append(slot)
            except BaseException:
                for slot in held:
                    slot.release()
                raise
            finally:
                with self.lock:
                    self.queued.discard(run_id)
            return held

        def execute() -> None:
            nonlocal held_reservation
            result: dict | None = None
            try:
                with (
                    run_observation(run_id, question, request.path, request.budget) as observation,
                    call_budget(deadline, lambda: run_id in self.stop_requested),
                    model_events(
                        lambda action, detail: sink(
                            TraceRow(
                                atS=round(monotonic() - run_started, 2),
                                action=action,
                                returned=detail,
                            )
                        )
                    ),
                ):
                    result = self._execute(request, question, run_id, sink, acquire, doc)
                    if observation is not None:
                        observation.update(
                            output={
                                "status": result["run"].status,
                                "triple_count": sum(
                                    len(item.triples) for item in result["run"].readings
                                ),
                                "claim_count": sum(
                                    len(item.claims) for item in result["run"].readings
                                ),
                            }
                        )
                with self.lock:
                    trace = list(self.events[run_id])
                # The saved trace is exactly what streamed live, including
                # model-call start/finish rows.
                result["artifacts"]["trace.json"] = json.dumps(
                    [item.model_dump(by_alias=True) for item in trace], indent=2
                ).encode()
                run = result["run"].model_copy(
                    update={
                        "evaluation_mode": request.evaluation_mode,
                        "snapshot": request.snapshot,
                        "parent_run_id": request.parent_run_id,
                        "run_input": run_input,
                        "settings": recorded_settings,
                        "trace": trace,
                        "prompt_hashes": {**result["run"].prompt_hashes, **prompt_versions(trace)},
                    }
                )
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
                    self.events[run_id].append(
                        TraceRow(
                            atS=0.0,
                            action="Run stopped" if stopped else "Run deadline exceeded",
                            returned="stopped by user"
                            if stopped
                            else f"exceeded {run_deadline_seconds(request.path, request.budget):.0f}s",
                        )
                    )
                self._finalize_terminal(
                    request,
                    question,
                    run_id,
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
                    request,
                    question,
                    run_id,
                    status="incomplete",
                    reason="internal_error",
                    message=f"Run failed unexpectedly: {type(exc).__name__}",
                )
            finally:
                if (self.artifacts.root / run_id / "run.json").exists():
                    self._pending_path(run_id).unlink(missing_ok=True)
                if held_reservation is not None:
                    for slot in held_reservation:
                        slot.release()
                    held_reservation = None
                with self.lock:
                    self.running.discard(run_id)
                    self.stop_requested.discard(run_id)

        threading.Thread(target=execute, name=f"run-{run_id}", daemon=True).start()
        return run_id

    def _execute(self, request: RunRequest, question: str, run_id: str, sink, acquire,
                 doc_file: Path | None = None) -> dict:
        """Run one request, holding model slots only while research runs. ``doc_file``: the evidence
        document that is the run's input (run input "evidence_doc"), else None."""

        def given(runner: Baseline20) -> evidence_docs.EvidenceDoc | None:
            if doc_file is None:
                return None
            return evidence_docs.load(doc_file.parent.parent, question, runner.corpus.by_passage)

        if request.path == "external" and not self.custom_runner:
            held = acquire(*run_slots(request.origin))
            try:
                external = self._external_runner(sink)
                external.evidence_doc = given(external)
                return external.run(question, request.budget, run_id)
            finally:
                for slot in held:
                    slot.release()
        if request.path != "agentic" or self.custom_runner:
            runner = (
                self.runner_factory(sink)
                if self.custom_runner
                else self._runner(sink, request.path == "baseline-enhanced")
            )
            kwargs = {"concept": request.concept} if request.concept else {}
            if doc_file is not None:
                runner.evidence_doc = given(runner)
            held = acquire(*run_slots(request.origin))
            try:
                return runner.run(question, request.budget, run_id, **kwargs)
            finally:
                for slot in held:
                    slot.release()

        # Granite handles the MCP tool loop; the configured heavy model handles
        # concept interpretation, structured extraction, and refinement.
        config = settings()
        agent_timeout = (
            config.budget_agentic_slow_seconds if request.budget == "slow"
            else config.budget_agentic_fast_seconds
        )
        agent_model = OpenAICompatibleModel(model=config.model_agent, timeout_s=agent_timeout)
        high_model = OpenAICompatibleModel(model=config.model_high, timeout_s=agent_timeout)
        baseline_model = config.model_small
        refinement_model = OpenAICompatibleModel(model=config.model_judge, timeout_s=agent_timeout)
        baseline = self._runner(
            sink,
            True,
            model_name=baseline_model,
            timeout_s=agent_timeout,
            refinement_model=refinement_model,
            schema_model=high_model,
            max_passages=config.agentic_max_passages,
        )
        runner = AgenticResearch(
            baseline,
            sink,
            agent_model=agent_model,
            high_model=high_model,
            **({"evidence_doc": doc} if (doc := given(baseline)) else {}),
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
            sink(
                TraceRow(
                    atS=0.0, action="Concept auto-selected for evaluation", returned=concept.label
                )
            )
        held = acquire(*run_slots(request.origin))
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
                request,
                request.question.strip(),
                run_id,
                status="incomplete",
                reason="interrupted",
                message="The API stopped while this run was active.",
            )
            if (marker.parent / "run.json").exists():
                marker.unlink(missing_ok=True)
            LOGGER.warning("Marked interrupted run %s as incomplete", run_id)

    def _finalize_terminal(
        self,
        request: RunRequest,
        question: str,
        run_id: str,
        *,
        status: str,
        reason: str | None,
        message: str,
    ) -> None:
        with self.lock:
            trace = list(self.events.get(run_id, []))
        run = Run(
            id=run_id,
            startedAt=datetime.now(UTC),
            durationS=0.0,
            question=question,
            path=request.path,
            budget=request.budget,
            status=status,
            reason=reason,
            trace=trace,
            concept=request.concept,
            parentRunId=request.parent_run_id,
            evaluationMode=request.evaluation_mode,
            snapshot=request.snapshot,
            runInput=request.run_input or "corpus",
            artifacts=list(required_artifacts(request.path)),
        )
        try:
            self.artifacts.finalize(
                run,
                {
                    "answer.md": message.encode(),
                    "trace.json": json.dumps(
                        [item.model_dump(by_alias=True) for item in trace]
                    ).encode(),
                },
            )
        except Exception:
            LOGGER.exception("Could not persist terminal state for run %s", run_id)

    def list(self) -> list[Run]:
        live = self.artifacts.list_runs()
        if not settings().include_fixtures:
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
                    run.concepts[index]
                    if index < len(run.concepts)
                    else ConceptChoice(
                        label=reading.label, gloss=reading.gloss, classes=reading.classes
                    )
                )
                # The question is kept verbatim; the concept is passed as
                # structured input so routing, matching and gold lookup all
                # see the user's actual question.
                child_id = self.start(
                    RunRequest(
                        question=run.question,
                        path=run.path,
                        budget=run.budget,
                        concept=concept,
                        parent_run_id=run.id,
                        evaluation_mode=run.evaluation_mode,
                        snapshot=run.snapshot,
                    )
                )
                existing = self.picks[run_id] = (child_id, index, concept)
        child_id, picked, concept = existing
        return run.model_copy(
            update={
                "id": child_id,
                "started_at": datetime.now(UTC),
                "duration_s": 0.0,
                "chosen_reading": picked,
                "concept": concept,
                "parent_run_id": run.id,
            }
        )

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


def change_ui_settings(runtime: RunRuntime, changes: dict[str, str | None]) -> dict[str, str]:
    """Apply Settings-page changes (validated; None restores the earlier value) and keep them in the file.
    Refused while a run is active: a run reads its settings throughout, and must not see two versions."""
    with runtime.lock:
        if runtime.running:
            raise RuntimeError(f"{len(runtime.running)} run(s) active: change settings when they have finished")
        runtime.ui_overrides = ui.apply(changes, runtime.ui_originals)
    ui.save_file(runtime.ui_file, runtime.ui_overrides)
    return runtime.ui_overrides
