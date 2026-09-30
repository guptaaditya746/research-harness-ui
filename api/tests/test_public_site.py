"""The password-protected site that serves the full UI and forwards /api to the harness API."""

from __future__ import annotations

import base64
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from public_site import create_app, users_from_env


def auth(user: str, password: str) -> dict[str, str]:
    return {"Authorization": "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()}


@pytest.fixture
def site(tmp_path: Path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<div id=root></div>")
    (dist / "assets" / "app.js").write_text("console.log(1)")
    (tmp_path / "secret.txt").write_text("outside the build")
    seen: list[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/runs/r-1/events":
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  content=b"event: trace\ndata: {}\n\n")
        return httpx.Response(200, json={"path": request.url.path, "query": request.url.query.decode(),
                                         "body": request.content.decode()})

    app = create_app({"ada": "correct horse battery"}, "http://api", dist, httpx.MockTransport(upstream))
    return TestClient(app), seen


def test_everything_needs_the_login(site) -> None:
    client, seen = site
    for path in ("/", "/assets/app.js", "/api/settings"):
        response = client.get(path)
        assert response.status_code == 401 and response.headers["www-authenticate"].startswith("Basic")
    assert client.get("/", headers=auth("ada", "wrong password here")).status_code == 401
    assert client.get("/", headers=auth("bob", "correct horse battery")).status_code == 401   # named users only
    assert seen == []


def test_the_ui_and_the_api_behind_the_login(site) -> None:
    client, seen = site
    headers = auth("ada", "correct horse battery")
    assert client.get("/", headers=headers).text == "<div id=root></div>"
    assert client.get("/runs", headers=headers).text == "<div id=root></div>"             # app pages
    assert client.get("/assets/app.js", headers=headers).text == "console.log(1)"
    assert client.get("/../secret.txt", headers=headers).text == "<div id=root></div>"     # never outside dist
    body = client.post("/api/runs?x=1", headers=headers, json={"question": "q"}).json()
    assert body == {"path": "/runs", "query": "x=1", "body": '{"question":"q"}'}
    assert "authorization" not in seen[-1].headers                                          # not forwarded
    events = client.get("/api/runs/r-1/events", headers=headers)
    assert events.headers["content-type"].startswith("text/event-stream") and b"event: trace" in events.content
    assert events.headers["x-frame-options"] == "DENY"


def test_repeated_failed_logins_are_locked_out(site) -> None:
    client, _ = site
    for _ in range(10):
        client.get("/", headers={**auth("ada", "nope nope nope"), "cf-connecting-ip": "203.0.113.9"})
    blocked = client.get("/", headers={**auth("ada", "correct horse battery"), "cf-connecting-ip": "203.0.113.9"})
    assert blocked.status_code == 429
    other = client.get("/", headers={**auth("ada", "correct horse battery"), "cf-connecting-ip": "198.51.100.4"})
    assert other.status_code == 200


def test_the_site_refuses_to_start_without_a_strong_password(monkeypatch) -> None:
    monkeypatch.delenv("SITE_USERS", raising=False)
    monkeypatch.delenv("SITE_PASSWORD", raising=False)
    with pytest.raises(RuntimeError, match="SITE_PASSWORD"):
        users_from_env()
    monkeypatch.setenv("SITE_PASSWORD", "short")
    with pytest.raises(RuntimeError, match="at least 12"):
        users_from_env()
    monkeypatch.setenv("SITE_USERS", "ada:correct horse battery,prof:another long secret")
    assert users_from_env() == {"ada": "correct horse battery", "prof": "another long secret"}
