from datetime import UTC, datetime
from pathlib import Path

import main
from fastapi.testclient import TestClient
from harness_schema.models import Metrics, Reading, Run, TraceRow, Triple, required_artifacts
from main import RunRuntime, app


def test_fixture_runs_are_contract_valid() -> None:
    client = TestClient(app)
    runs = client.get("/runs").json()
    assert len(runs) == 4
    assert {run["status"] for run in runs} == {"answered", "awaiting_reading", "incomplete"}
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
