"""Heavy-model load comes from the thesis-llm-setup status API."""

from __future__ import annotations

import httpx
from harness_agent import AgenticResearch


class _Response:
    def __init__(self, payload: dict | None = None, text: str = "", status: int = 200) -> None:
        self._payload, self.text, self.status_code = payload, text, status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=None)  # type: ignore[arg-type]

    def json(self) -> dict:
        return self._payload or {}


def _status(state: str, waiting: int) -> dict:
    return {"overall": state, "models": {"heavy-model": {"state": state, "waiting": waiting}}}


def test_waiting_requests_are_read_from_the_status_api(monkeypatch) -> None:
    seen = []
    monkeypatch.setenv("LLM_STATUS_URL", "http://status/v1/status")
    monkeypatch.setattr(httpx, "get", lambda url, **_: seen.append(url) or _Response(_status("busy", 3)))
    assert AgenticResearch._heavy_queue_depth() == 3
    assert seen == ["http://status/v1/status"]


def test_saturated_heavy_counts_as_at_least_the_threshold(monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_HEAVY_QUEUE_THRESHOLD", "8")
    monkeypatch.setattr(httpx, "get", lambda url, **_: _Response(_status("saturated", 1)))
    assert AgenticResearch._heavy_queue_depth() == 8


def test_unavailable_heavy_reports_unknown_load(monkeypatch) -> None:
    monkeypatch.setattr(httpx, "get", lambda url, **_: _Response(_status("unavailable", 0)))
    assert AgenticResearch._heavy_queue_depth() is None


def test_falls_back_to_vllm_metrics_when_status_api_is_down(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_HEAVY_API_BASE", "http://127.0.0.1:8002/v1")

    def get(url, **_):
        if url.endswith("/v1/status"):
            raise httpx.ConnectError("down")
        return _Response(text='vllm:num_requests_waiting{engine="0"} 5.0\n')

    monkeypatch.setattr(httpx, "get", get)
    assert AgenticResearch._heavy_queue_depth() == 5
