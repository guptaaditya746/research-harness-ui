import time
from datetime import UTC, datetime
from pathlib import Path

import main
from fastapi.testclient import TestClient
from harness_schema.models import Metrics, Reading, Run, TraceRow, Triple, required_artifacts
from main import RunRuntime, app


def test_fixture_runs_are_contract_valid() -> None:
    client = TestClient(app)
    runs = client.get("/runs").json()
    fixtures = {run.id for run in main.fixture_runs()}
    fixture_rows = [run for run in runs if run["id"] in fixtures]
    assert len(fixture_rows) == len(fixtures) == 4
    assert {run.status for run in main.fixture_runs()} == {"answered", "awaiting_reading", "incomplete"}
    assert client.get("/runs/r-0412").json()["chosenReading"] == 0
    assert client.get("/runs/missing").status_code == 404


def test_artifact_preview_is_accessible() -> None:
    client = TestClient(app)
    result = client.get("/runs/r-0412/artifacts/query.rq")
    assert result.status_code == 200
    assert "SELECT" in result.json()["content"]


def test_reading_selection_starts_a_new_criterion_run(tmp_path: Path, monkeypatch) -> None:
    test_runtime = RunRuntime(tmp_path / "artifacts", tmp_path / "datasets")
    option = Reading(
        label="Most distinct papers", gloss="Count distinct works.",
        classes=[], axioms=[], sparql="", answer="", triples=[], metrics=Metrics(),
    )
    original = Run(
        id="r-ambiguous", startedAt=datetime.now(UTC), durationS=0.0,
        question="Who is the best author in the corpus?", path="baseline-enhanced",
        budget="slow", status="awaiting_reading", readings=[option],
        artifacts=list(required_artifacts("baseline-enhanced")),
    )
    test_runtime.artifacts.finalize(original, {})
    submitted = []
    monkeypatch.setattr(test_runtime, "start", lambda request: submitted.append(request) or "r-selected")

    selected = test_runtime.pick_reading("r-ambiguous", 0)

    assert selected.id == "r-selected"
    assert selected.status == "awaiting_reading"
    assert selected.chosen_reading == 0
    # The question is never rewritten; the chosen concept travels alongside it.
    assert submitted[0].question == "Who is the best author in the corpus?"
    assert submitted[0].concept.label == "Most distinct papers"
    assert submitted[0].parent_run_id == "r-ambiguous"
    assert selected.parent_run_id == "r-ambiguous"
    # A second pick (double click) does not start another run.
    assert test_runtime.pick_reading("r-ambiguous", 0).id == "r-selected"
    assert len(submitted) == 1
    assert test_runtime.get("r-ambiguous").status == "awaiting_reading"


class FakeRunner:
    def __init__(self, sink) -> None:
        self.sink = sink

    def run(self, question: str, budget: str, run_id: str):
        row = TraceRow(atS=0.01, action="Retrieving top passages", returned="1 passage")
        self.sink(row)
        reading = Reading(
            label="Introduced method",
            gloss="Paper proposing the method",
            classes=["Paper", "Claim"],
            properties=["claims"],
            axioms=[("claims", "Paper", "Claim")],
            sparql="SELECT ?paper WHERE { ?paper <https://harness.local/schema#claims> ?claim . }",
            answer="p4 introduced Video Prediction Policy [p4-abstract].",
            triples=[
                Triple(
                    s="p4",
                    p="claims",
                    o="Video Prediction Policy",
                    support="1 passage",
                    evidence="we propose Video Prediction Policy",
                    source="p4-abstract",
                    tripleId="t-001",
                )
            ],
            metrics=Metrics(),
            status="answered",
        )
        run = Run(
            id=run_id,
            startedAt=datetime.now(UTC),
            durationS=0.02,
            question=question,
            path="baseline",
            budget=budget,
            status="answered",
            readings=[reading],
            chosenReading=0,
            trace=[row],
            artifacts=list(required_artifacts("baseline")),
        )
        artifacts = {name: b"test" for name in required_artifacts("baseline") if name != "run.json"}
        return {"run": run, "artifacts": artifacts}


def test_live_run_sse_rating_promotion_and_editor(tmp_path: Path, monkeypatch) -> None:
    test_runtime = RunRuntime(
        tmp_path / "artifacts",
        tmp_path / "datasets",
        runner_factory=lambda sink: FakeRunner(sink),
    )
    monkeypatch.setattr(main, "runtime", test_runtime)
    client = TestClient(app)

    created = client.post(
        "/runs",
        json={"question": "Which paper introduced VPP?", "path": "baseline", "budget": "fast"},
    )
    assert created.status_code == 202
    run_id = created.json()["id"]
    events = client.get(f"/runs/{run_id}/events")
    assert "event: trace" in events.text
    assert "event: done" in events.text
    assert client.get(f"/runs/{run_id}").json()["status"] == "answered"
    evaluated = client.post(
        f"/runs/{run_id}/evaluate",
        json={"gold_triples": [{"s": "p4", "p": "claims", "o": "Video Prediction Policy"}]},
    )
    assert evaluated.status_code == 200
    assert evaluated.json()["chosen"]["f1"] == 1.0

    assert client.post(f"/runs/{run_id}/promote").status_code == 409
    rated = client.post(f"/runs/{run_id}/rating", json={"rating": "useful"})
    assert rated.json()["rating"] == "useful"
    candidate = client.post(f"/runs/{run_id}/promote").json()
    assert candidate["status"] == "draft"
    edited = client.put(
        f"/datasets/candidates/{candidate['id']}",
        json={"notes": "Reviewed", "status": "approved"},
    )
    assert edited.json()["notes"] == "Reviewed"
    assert len(client.get("/datasets/candidates").json()) == 1


def test_benchmark_endpoint_aggregates_approved_question(tmp_path: Path, monkeypatch) -> None:
    test_runtime = RunRuntime(
        tmp_path / "artifacts",
        tmp_path / "datasets",
        runner_factory=lambda sink: FakeRunner(sink),
    )
    monkeypatch.setattr(main, "runtime", test_runtime)
    client = TestClient(app)
    created = client.post(
        "/runs",
        json={"question": "Which paper introduced VPP?", "path": "baseline", "budget": "fast"},
    )
    run_id = created.json()["id"]
    for _ in range(100):
        if client.get(f"/runs/{run_id}").json().get("status") == "answered":
            break
        time.sleep(0.01)
    client.post(f"/runs/{run_id}/rating", json={"rating": "useful"})
    candidate = client.post(f"/runs/{run_id}/promote").json()
    edited = client.put(
        f"/datasets/candidates/{candidate['id']}",
        json={"category": "metadata", "gold_version": "v1", "status": "approved"},
    )
    assert edited.status_code == 200
    assert edited.json()["status"] == "approved"
    response = client.get("/benchmark")
    assert response.status_code == 200
    assert response.json()["questions"][0]["paths"][0]["f1"]["mean"] == 1.0

    degraded = test_runtime.get(run_id).model_copy(
        update={"reason": "agentic_degraded: heavy_model_busy: 93 requests waiting"}
    )
    test_runtime.artifacts.update_run(degraded)
    response = client.get("/benchmark")
    path_row = response.json()["questions"][0]["paths"][0]
    assert path_row["completed"] == 0
    assert path_row["failed"] == 1
    assert path_row["f1"]["mean"] is None


def test_benchmark_execute_validates_and_launches_repetitions(tmp_path: Path, monkeypatch) -> None:
    test_runtime = RunRuntime(
        tmp_path / "artifacts",
        tmp_path / "datasets",
        runner_factory=lambda sink: FakeRunner(sink),
    )
    monkeypatch.setattr(main, "runtime", test_runtime)
    client = TestClient(app)
    response = client.post(
        "/benchmark/execute",
        json={"question": "q", "path": "baseline", "budget": "fast", "repetitions": 3},
    )
    assert response.status_code == 202
    assert response.json()["evaluation_mode"] is True
    assert len(response.json()["ids"]) == 3


class SlowRunner:
    """Emits a trace row every 20 ms for ~10 s, like a long research run."""

    def __init__(self, sink) -> None:
        self.sink = sink

    def run(self, question: str, budget: str, run_id: str):
        for step in range(500):
            self.sink(TraceRow(atS=step * 0.02, action="Working", returned=str(step)))
            time.sleep(0.02)
        raise AssertionError("run should have been cancelled")


def _wait_terminal(runtime: RunRuntime, run_id: str, limit: float = 5.0) -> Run:
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        if run_id not in runtime.running:
            return runtime.get(run_id)
        time.sleep(0.02)
    raise AssertionError("run did not reach a terminal state")


def test_stop_cancels_a_running_run_and_frees_its_slot(tmp_path: Path) -> None:
    runtime = RunRuntime(tmp_path / "artifacts", tmp_path / "datasets", runner_factory=SlowRunner)
    run_id = runtime.start(main.RunRequest(question="Q?", path="baseline", budget="fast"))
    time.sleep(0.1)
    started = time.monotonic()
    runtime.stop(run_id)
    run = _wait_terminal(runtime, run_id)
    assert time.monotonic() - started < 1.0
    assert run.status == "stopped"
    assert run.trace[-1].action == "Run stopped"
    # Both model slots are free again.
    assert main.MODEL_SLOT.acquire(timeout=0.1)
    main.MODEL_SLOT.release()


def test_run_deadline_ends_run_as_incomplete(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_RUN_DEADLINE_FAST_S", "0.2")
    runtime = RunRuntime(tmp_path / "artifacts", tmp_path / "datasets", runner_factory=SlowRunner)
    run_id = runtime.start(main.RunRequest(question="Q?", path="baseline", budget="fast"))
    run = _wait_terminal(runtime, run_id)
    assert run.status == "incomplete"
    assert run.reason == "deadline_exceeded"


def test_corrupt_run_file_does_not_hide_other_runs(tmp_path: Path) -> None:
    runtime = RunRuntime(tmp_path / "artifacts", tmp_path / "datasets")
    good = Run(id="r-good", startedAt=datetime.now(UTC), durationS=0.0, question="Q?",
               path="baseline", budget="fast", status="stopped",
               artifacts=list(required_artifacts("baseline")))
    runtime.artifacts.finalize(good, {})
    bad = tmp_path / "artifacts" / "r-bad"
    bad.mkdir(parents=True)
    (bad / "run.json").write_text("{not json")
    assert [run.id for run in runtime.artifacts.list_runs()] == ["r-good"]


def test_malformed_gold_claims_are_rejected_at_write_time(tmp_path: Path) -> None:
    runtime = RunRuntime(tmp_path / "artifacts", tmp_path / "datasets")
    runtime.gold._write([{"id": "gold-1", "question": "Q?", "status": "draft"}])
    try:
        runtime.gold.update("gold-1", {"gold_claims": [{"id": "c1"}]})
    except ValueError as exc:
        assert "gold_claims[0]" in str(exc)
    else:
        raise AssertionError("malformed gold claim was accepted")


def test_runs_active_during_an_api_restart_are_closed_as_interrupted(tmp_path: Path) -> None:
    first = RunRuntime(tmp_path / "artifacts", tmp_path / "datasets")
    first._mark_pending("r-lost", main.RunRequest(question="Q?", path="agentic", budget="slow"))
    restarted = RunRuntime(tmp_path / "artifacts", tmp_path / "datasets")
    run = restarted.get("r-lost")
    assert run.status == "incomplete"
    assert run.reason == "interrupted"
    assert not (tmp_path / "artifacts" / "r-lost" / "pending.json").exists()


def test_model_calls_respect_the_run_budget() -> None:
    from harness_extraction import ModelFailure, bounded_timeout, call_budget

    assert bounded_timeout(120) == 120
    with call_budget(time.monotonic() + 30, lambda: False):
        assert 28 < bounded_timeout(120) <= 30
    with call_budget(time.monotonic() + 0.5, lambda: False):
        try:
            bounded_timeout(120)
        except ModelFailure as exc:
            assert "deadline" in str(exc)
        else:
            raise AssertionError("call started after the deadline")
    with call_budget(time.monotonic() + 60, lambda: True):
        try:
            bounded_timeout(120)
        except ModelFailure as exc:
            assert "stopped" in str(exc)
        else:
            raise AssertionError("call started after stop")
