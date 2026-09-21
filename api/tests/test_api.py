from fastapi.testclient import TestClient
from main import app


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
