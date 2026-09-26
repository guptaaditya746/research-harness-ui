import json
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import app_state
import main
import runtime as runtime_module
from fastapi.testclient import TestClient
from harness_schema.models import Metrics, Reading, Run, TraceRow, Triple, required_artifacts
from main import RunRuntime, app


def test_fixture_runs_are_contract_valid() -> None:
    client = TestClient(app)
    runs = client.get("/runs").json()
    fixtures = {run.id for run in main.fixture_runs()}
    fixture_rows = [run for run in runs if run["id"] in fixtures]
    assert len(fixture_rows) == len(fixtures) == 4
    assert {run.status for run in main.fixture_runs()} == {
        "answered",
        "awaiting_reading",
        "incomplete",
    }
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
        label="Most distinct papers",
        gloss="Count distinct works.",
        classes=[],
        axioms=[],
        sparql="",
        answer="",
        triples=[],
        metrics=Metrics(),
    )
    original = Run(
        id="r-ambiguous",
        startedAt=datetime.now(UTC),
        durationS=0.0,
        question="Who is the best author in the corpus?",
        path="baseline-enhanced",
        budget="slow",
        status="awaiting_reading",
        readings=[option],
        artifacts=list(required_artifacts("baseline-enhanced")),
    )
    test_runtime.artifacts.finalize(original, {})
    submitted = []
    monkeypatch.setattr(
        test_runtime, "start", lambda request: submitted.append(request) or "r-selected"
    )

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


def test_agentic_routes_mcp_to_lite_and_quality_roles_to_heavy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("HARNESS_MODEL_AGENT", "lite-model")
    monkeypatch.setenv("HARNESS_MODEL_SMALL", "heavy-model")
    monkeypatch.setenv("HARNESS_MODEL_HIGH", "heavy-model")
    monkeypatch.setenv("HARNESS_MODEL_JUDGE", "heavy-model")
    test_runtime = RunRuntime(tmp_path / "artifacts", tmp_path / "datasets")
    models = []
    captured = {}

    class FakeModel:
        def __init__(self, model: str, timeout_s: float) -> None:
            self.model = model
            self.timeout_s = timeout_s
            models.append(self)

    class FakeAgenticResearch:
        def __init__(self, baseline, sink, agent_model, high_model) -> None:
            captured["agent_model"] = agent_model
            captured["high_model"] = high_model

        def propose(self, question: str, budget: str, run_id: str) -> dict:
            return {"run": SimpleNamespace(status="incomplete"), "marker": "proposal"}

    def fake_runner(sink, enhanced: bool = False, **kwargs):
        captured["runner_args"] = {"enhanced": enhanced, **kwargs}
        return object()

    monkeypatch.setattr(runtime_module, "OpenAICompatibleModel", FakeModel)
    monkeypatch.setattr(runtime_module, "AgenticResearch", FakeAgenticResearch)
    monkeypatch.setattr(test_runtime, "_runner", fake_runner)

    result = test_runtime._execute(
        main.RunRequest(question="Q?", path="agentic", budget="fast"),
        "Q?",
        "r-routing",
        lambda _: None,
        lambda _: [],
    )

    assert result["marker"] == "proposal"
    assert captured["agent_model"].model == "lite-model"
    assert captured["high_model"].model == "heavy-model"
    runner_args = captured["runner_args"]
    assert runner_args["model_name"] == "heavy-model"
    assert runner_args["schema_model"].model == "heavy-model"
    assert runner_args["refinement_model"].model == "heavy-model"


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
    monkeypatch.setattr(app_state, "runtime", test_runtime)
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
    monkeypatch.setattr(app_state, "runtime", test_runtime)
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
    body = response.json()
    path_row = body["questions"][0]["paths"][0]
    assert path_row["f1"]["mean"] == 1.0
    assert path_row["levels"]["strict"]["f1"]["mean"] == 1.0
    assert path_row["levels"]["soft"]["f1"]["mean"] == 1.0
    assert path_row["judge_pending"] == 0
    everything = next(row for row in body["by_complexity"]["rows"] if row["complexity"] == "all")
    assert everything["questions"] == 1 and everything["paths"]["baseline"]["soft"]["mean"] == 1.0

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
    monkeypatch.setattr(app_state, "runtime", test_runtime)
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
    good = Run(
        id="r-good",
        startedAt=datetime.now(UTC),
        durationS=0.0,
        question="Q?",
        path="baseline",
        budget="fast",
        status="stopped",
        artifacts=list(required_artifacts("baseline")),
    )
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


def test_question_library_merges_curated_gold_and_recent(tmp_path: Path, monkeypatch) -> None:
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    (datasets / "question_matrix_20.json").write_text(
        '[{"id": "q01", "category": "metadata", "type": "paper_identification", '
        '"question": "Which paper introduced Video Prediction Policy?"}]'
    )
    test_runtime = RunRuntime(tmp_path / "artifacts", datasets)
    test_runtime.gold._write(
        [
            {
                "id": "gold-1",
                "question": "Which paper introduced  Video Prediction Policy?",
                "status": "draft",
            },
            {"id": "gold-2", "question": "Who wrote FOCUS?", "status": "approved"},
        ]
    )
    monkeypatch.setenv("DATASETS_ROOT", str(datasets))
    monkeypatch.setenv("HARNESS_QUESTION_SETS", "question_matrix_20.json")
    monkeypatch.setattr(app_state, "runtime", test_runtime)
    only_sets = TestClient(app).get("/questions").json()["questions"]
    assert [item["question"] for item in only_sets] == ["Which paper introduced Video Prediction Policy?"]
    monkeypatch.setenv("HARNESS_QUESTION_EXTRAS", "true")
    body = TestClient(app).get("/questions").json()["questions"]
    assert [(item["question"], item["category"]) for item in body] == [
        ("Which paper introduced Video Prediction Policy?", "metadata"),
        ("Who wrote FOCUS?", "gold"),
    ]


def test_default_question_library_is_the_thirty_evaluation_questions(monkeypatch) -> None:
    monkeypatch.delenv("HARNESS_QUESTION_SETS", raising=False)
    monkeypatch.delenv("DATASETS_ROOT", raising=False)
    paths = main._question_sets()
    assert [path.name for path in paths] == ["question_library_2025.json"]
    nlp_rows = json.loads(paths[0].read_text())
    assert [row["id"] for row in nlp_rows] == [f"nlp{number:02d}" for number in range(1, 31)]
    assert len(nlp_rows) == 30
    assert len({row["question"] for row in nlp_rows}) == 30
    assert any(row["anchor"].startswith("2025.acl-long.") for row in nlp_rows)
    assert any(row["anchor"].startswith("2025.emnlp-main.") for row in nlp_rows)
    assert {row["category"] for row in nlp_rows} >= {"metadata", "aggregation", "synthesis"}
    corpus_rows = json.loads((paths[0].parent / "corpus_10.json").read_text())
    assert len(corpus_rows) == 10
    assert [row["id"] for row in corpus_rows] == [f"q{number:02d}" for number in range(1, 11)]
    assert all(row["expected"] == "answer" for row in corpus_rows)


def test_public_capacity_reservation_is_loopback_only_and_one_use(tmp_path: Path, monkeypatch) -> None:
    remote = TestClient(app, client=("198.51.100.8", 1234))
    assert remote.post("/internal/public/reservations").status_code == 403

    runtime = RunRuntime(
        tmp_path / "artifacts",
        tmp_path / "datasets",
        runner_factory=lambda sink: FakeRunner(sink),
    )
    monkeypatch.setattr(app_state, "runtime", runtime)
    client = TestClient(app, client=("127.0.0.1", 1234))
    response = client.post("/internal/public/reservations")
    assert response.status_code == 200
    ticket = response.json()["reservation"]
    assert client.post("/internal/public/reservations").status_code == 503

    started = client.post(
        "/runs",
        json={
            "question": "A public research question",
            "path": "baseline",
            "budget": "fast",
            "origin": "public",
            "deadline_seconds": 90,
        },
        headers={"X-Public-Reservation": ticket},
    )
    assert started.status_code == 202
    run_id = started.json()["id"]
    assert _wait_terminal(runtime, run_id).status == "answered"
    assert client.post(
        "/runs",
        json={"question": "Another public question", "path": "baseline", "budget": "fast", "origin": "public"},
        headers={"X-Public-Reservation": ticket},
    ).status_code == 503

    assert main.PUBLIC_MODEL_SLOT.acquire(blocking=False)
    main.PUBLIC_MODEL_SLOT.release()


def test_prompt_versions_are_collected_from_model_call_rows() -> None:
    from runtime import prompt_versions

    rows = [
        TraceRow(atS=0.1, action="Model call finished",
                 returned="heavy-model · schema inference · 3.2s · prompt extraction/schema_inference@v2:1a2b3c4d"),
        TraceRow(atS=0.2, action="Model call finished", returned="lite-model · agent tool loop turn 2"),
    ]
    assert prompt_versions(rows) == {"extraction/schema_inference": "v2:1a2b3c4d"}
