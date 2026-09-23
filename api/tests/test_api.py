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
    assert candidate["status"] == "candidate"
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
