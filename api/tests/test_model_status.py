"""GET /models/status relays the thesis-llm-setup status API and adds roles and the run queue."""

from __future__ import annotations

from typing import Self

import routes_status
from fastapi.testclient import TestClient
from main import app

DOCUMENT = {
    "overall": "busy", "updatedAt": "2026-09-27T10:00:00Z",
    "models": {
        "heavy-model": {"state": "busy", "running": 4, "waiting": 3, "kvCache": 0.83, "queueP95Seconds": 2.4},
        "lite-model": {"state": "healthy", "running": 1, "waiting": 0, "kvCache": 0.12},
        "qwen-embed": {"state": "unavailable"},
    },
}


class Response:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload


class Client:
    def __init__(self, *args, fail: bool = False, **kwargs) -> None:
        self.fail = fail

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args) -> None:
        return None

    async def get(self, url: str) -> Response:
        if self.fail:
            raise routes_status.httpx.ConnectError("refused")
        return Response(DOCUMENT)


def test_status_lists_each_model_with_load_queue_roles_and_the_run_queue(monkeypatch) -> None:
    monkeypatch.setenv("LLM_STATUS_URL", "http://status/v1/status")
    monkeypatch.setenv("HARNESS_MODEL_AGENT", "lite-model")
    monkeypatch.setattr(routes_status.httpx, "AsyncClient", Client)
    body = TestClient(app).get("/models/status").json()
    assert body["reachable"] is True and body["overall"] == "busy"
    heavy = next(item for item in body["models"] if item["alias"] == "heavy-model")
    assert (heavy["state"], heavy["waiting"], heavy["kvCache"]) == ("busy", 3, 0.83)
    assert "schema and answers" in heavy["roles"]
    lite = next(item for item in body["models"] if item["alias"] == "lite-model")
    assert lite["roles"] == ["agent"]
    assert next(item for item in body["models"] if item["alias"] == "qwen-embed")["state"] == "unavailable"
    assert set(body["runs"]) == {"active", "queued", "capacity"}


def test_unreachable_status_api_is_reported_not_raised(monkeypatch) -> None:
    monkeypatch.setenv("LLM_STATUS_URL", "http://status/v1/status")
    monkeypatch.setattr(routes_status.httpx, "AsyncClient", lambda *a, **k: Client(fail=True))
    body = TestClient(app).get("/models/status").json()
    assert body["reachable"] is False and "is unreachable" in body["detail"]
    assert body["models"] and all(item["state"] == "unknown" for item in body["models"])


def test_other_spellings_of_the_counts_and_provider_prefixed_aliases_are_read(monkeypatch) -> None:
    document = {"overall": "busy", "models": {
        "openai/heavy-model": {"state": "busy", "requestsRunning": 2, "queued": 5},
        "lite-model": {"state": "healthy", "running": 0, "waiting": 0},
        "Qwen/Qwen3-8B": {"state": "healthy", "running": 1, "waiting": 2},
    }}

    class Other(Client):
        async def get(self, url: str) -> Response:
            return Response(document)

    monkeypatch.setenv("LLM_STATUS_URL", "http://status/v1/status")
    monkeypatch.setenv("HARNESS_MODEL_HIGH", "openai/heavy-model")
    monkeypatch.setattr(routes_status.httpx, "AsyncClient", Other)
    body = TestClient(app).get("/models/status").json()
    heavy = [item for item in body["models"] if item["alias"] == "heavy-model"]
    assert len(heavy) == 1 and (heavy[0]["running"], heavy[0]["waiting"]) == (2, 5)
    assert "schema and answers" in heavy[0]["roles"]                          # one row, not two
    lite = next(item for item in body["models"] if item["alias"] == "lite-model")
    assert (lite["running"], lite["waiting"]) == (0, 0)                       # zero is a value, not missing
    assert next(item for item in body["models"] if item["alias"] == "Qwen/Qwen3-8B")["waiting"] == 2
