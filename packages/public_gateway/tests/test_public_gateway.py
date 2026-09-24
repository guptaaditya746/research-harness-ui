from fastapi.testclient import TestClient
from harness_public_gateway.app import Config, TurnstileVerifier, create_app, project_run


class FakeVerifier:
    def __init__(self, ok=True):
        self.ok = ok

    def verify(self, token, remote_ip=None):
        return self.ok


class FakePrivate:
    def __init__(self):
        self.calls = []

    def reserve(self):
        return "reservation-token"

    def start(self, question, path, budget, deadline_s, reservation):
        self.calls.append((question, path, budget, deadline_s, reservation))
        return "r-12345678"

    def get(self, run_id):
        return {"status": "incomplete", "reason": "secret stack trace"}

    def events(self, run_id):
        raise AssertionError("gateway must not proxy events")

    def can_start(self):
        return True

    def ready(self):
        return True


def test_projection_fails_closed_and_never_returns_private_fields():
    run = {
        "id": "private-id",
        "status": "incomplete",
        "reason": "/private/path secret token",
        "trace": [{"returned": "raw prompt"}],
        "artifacts": ["trace.json"],
    }
    out = project_run(run)
    assert out["state"] == "incomplete" and out["answer"] is None
    assert "/private/path" not in str(out) and "raw prompt" not in str(out)


def test_private_origin_loopback_only_and_no_path_input():
    import pytest
    from harness_public_gateway.app import LoopbackPrivateClient

    with pytest.raises(ValueError):
        LoopbackPrivateClient("https://example.com")


def test_turnstile_fails_closed_and_bypass_configuration_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        TurnstileVerifier("secret", True, True)
    assert not FakeVerifier(False).verify("token")


def test_only_public_routes_and_opaque_mapping():
    private = FakePrivate()
    app = create_app(
        private,
        FakeVerifier(),
        Config(
            origins=["https://public.example"],
            identity_salt="salt",
            verify_turnstile=False,
            dev_bypass=True,
            environment="development",
        ),
    )
    c = TestClient(app)
    assert c.get("/runs").status_code == 404
    resp = c.post(
        "/v1/public/runs",
        json={
            "question": "A question with enough characters",
            "path": "fast",
            "turnstileToken": "ok",
        },
        headers={"Origin": "https://public.example"},
    )
    assert resp.status_code == 202
    public_id = resp.json()["publicRunId"]
    assert public_id != "private-123" and "private-123" not in resp.text
    assert c.get(f"/v1/public/runs/{public_id}/../../settings").status_code in {404, 307, 400}


def test_queue_and_client_limits_do_not_start_private_work():
    import pytest
    from fastapi import HTTPException
    from harness_public_gateway.app import Admission

    private = FakePrivate()
    admission = Admission(private, max_queue=1, per_day=1)
    admission.stop.set()  # Hold the test queue without starting private work.
    admission.enqueue("Question one is long enough", "fast", "same-client")
    with pytest.raises(HTTPException) as full:
        admission.enqueue("Question two is long enough", "fast", "other-client")
    assert full.value.status_code == 429
    with pytest.raises(HTTPException) as quota:
        admission.enqueue("Question three is long enough", "fast", "same-client")
    assert quota.value.status_code == 429
    assert private.calls == []


def test_single_active_run_per_client():
    import pytest
    from fastapi import HTTPException
    from harness_public_gateway.app import Admission

    admission = Admission(FakePrivate())
    admission.stop.set()
    admission.enqueue("Question one is long enough", "fast", "same-client")
    with pytest.raises(HTTPException) as error:
        admission.enqueue("Question two is long enough", "fast", "same-client")
    assert error.value.status_code == 429


def test_public_projection_requires_claim_backed_triple_provenance():
    run = {
        "status": "answered",
        "chosenReading": 0,
        "readings": [
            {
                "answer": "Supported answer.",
                "label": "safe",
                "gloss": "safe",
                "answerClaimIds": ["claim-private"],
                "claims": [
                    {
                        "id": "claim-private",
                        "kind": "ResultClaim",
                        "fields": {
                            "result": {"value": "x", "sourceIds": ["src-local"], "quote": "secret"}
                        },
                        "sourceIds": ["src-local"],
                        "evidence": {"src-local": "Excerpt supports x."},
                        "evidenceAnchors": [
                            {
                                "id": "anchor-1",
                                "kind": "passage",
                                "text": "Passage excerpt",
                                "locator": {"path": "/private/corpus"},
                                "sourceId": "src-local",
                            }
                        ],
                    }
                ],
                "triples": [
                    {
                        "s": "A",
                        "p": "has result",
                        "o": "x",
                        "source": "src-local",
                        "evidence": "Excerpt supports x.",
                        "support": "supported",
                        "claimId": "claim-private",
                    },
                    {
                        "s": "bad",
                        "p": "has",
                        "o": "value",
                        "source": "other",
                        "evidence": "unsupported",
                    },
                ],
                "graph": {
                    "nodes": [
                        {
                            "id": "/private/id",
                            "label": "A",
                            "kind": "entity",
                            "iri": "http://localhost:8000",
                        }
                    ],
                    "edges": [],
                },
            }
        ],
    }
    out = project_run(run)
    assert out["state"] == "answered" and len(out["triples"]) == 1
    rendered = str(out)
    assert (
        "private/corpus" not in rendered
        and "localhost" not in rendered
        and "src-local" not in rendered
    )
    assert "quote" not in rendered and "claim-private" not in rendered


def test_turnstile_failure_rejects_without_private_start():
    private = FakePrivate()
    app = create_app(
        private,
        FakeVerifier(False),
        Config(
            origins=[],
            identity_salt="salt",
            verify_turnstile=False,
            dev_bypass=True,
            environment="development",
        ),
    )
    response = TestClient(app).post(
        "/v1/public/runs",
        json={"question": "Enough characters here", "path": "fast", "turnstileToken": "bad"},
    )
    assert response.status_code == 400 and private.calls == []


def test_public_sse_does_not_project_private_trace():
    private = FakePrivate()
    app = create_app(
        private,
        FakeVerifier(),
        Config(
            identity_salt="salt", verify_turnstile=False, dev_bypass=True, environment="development"
        ),
    )
    admission = app.state.admission
    admission.stop.set()
    entry = admission.enqueue("Question is sufficiently long", "fast", "client")
    entry.progress = [{"message": "Research is in progress."}]
    entry.state = "incomplete"
    response = TestClient(app).get(f"/v1/public/runs/{entry.public_id}/events")
    assert "Research is in progress." in response.text
    assert "trace" not in response.text and "private-123" not in response.text


def test_wildcard_cors_origin_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        Config(origins=["*"])


def test_turnstile_posts_secret_and_token_to_server_verifier(monkeypatch):
    import httpx
    from harness_public_gateway.app import TurnstileVerifier

    seen = {}

    def fake_post(url, data, timeout):
        seen.update(url=url, data=data, timeout=timeout)
        return httpx.Response(200, json={"success": True})

    monkeypatch.setattr(httpx, "post", fake_post)
    verifier = TurnstileVerifier("private-secret", enabled=True)
    assert verifier.verify("one-use-token")
    assert seen["data"] == {"secret": "private-secret", "response": "one-use-token"}
    assert "one-use-token" not in repr(verifier.__dict__)


def test_model_saturation_rejects_before_private_start():
    import threading

    from harness_public_gateway.app import Admission

    class Saturated(FakePrivate):
        def __init__(self):
            super().__init__()
            self.done = threading.Event()

        def can_start(self):
            return False

        def get(self, run_id):
            return {"status": "incomplete"}

    private = Saturated()
    admission = Admission(private)
    entry = admission.enqueue("Question is sufficiently long", "fast", "client")
    for _ in range(50):
        if entry.state == "rejected":
            break
        threading.Event().wait(0.02)
    assert entry.state == "rejected" and entry.failure == "service busy"
    assert private.calls == []


def test_private_api_unavailable_returns_friendly_503():
    class Offline(FakePrivate):
        def ready(self):
            raise OSError("internal URL and token")

    app = create_app(
        Offline(),
        FakeVerifier(),
        Config(
            identity_salt="salt", verify_turnstile=False, dev_bypass=True, environment="development"
        ),
    )
    response = TestClient(app).post(
        "/v1/public/runs",
        json={"question": "Question is sufficiently long", "path": "fast", "turnstileToken": "ok"},
    )
    assert response.status_code == 503
    assert "internal URL" not in response.text and "token" not in response.text


def test_cors_preflight_accepts_only_exact_public_origin():
    app = create_app(
        FakePrivate(),
        FakeVerifier(),
        Config(
            origins=["https://public.example"],
            identity_salt="salt",
            verify_turnstile=False,
            dev_bypass=True,
            environment="development",
        ),
    )
    client = TestClient(app)
    allowed = client.options(
        "/v1/public/runs",
        headers={
            "Origin": "https://public.example",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    denied = client.options(
        "/v1/public/runs",
        headers={
            "Origin": "https://attacker.example",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert allowed.headers["access-control-allow-origin"] == "https://public.example"
    assert "access-control-allow-origin" not in denied.headers


def test_validation_errors_never_echo_turnstile_token():
    app = create_app(
        FakePrivate(),
        FakeVerifier(),
        Config(
            identity_salt="salt", verify_turnstile=False, dev_bypass=True, environment="development"
        ),
    )
    token = "x" * 3000
    response = TestClient(app).post(
        "/v1/public/runs",
        json={"question": "Question is long enough", "path": "fast", "turnstileToken": token},
    )
    assert response.status_code == 422
    assert token not in response.text
    assert "invalid_request" in response.text


def test_development_bypass_is_rejected_in_production():
    import pytest

    with pytest.raises(ValueError, match="forbidden outside development"):
        create_app(
            FakePrivate(),
            FakeVerifier(),
            Config(verify_turnstile=False, dev_bypass=True, environment="production"),
        )


def test_private_client_uses_fixed_routes_and_public_marker(monkeypatch):
    import httpx
    import pytest
    from harness_public_gateway.app import LoopbackPrivateClient

    requests = []

    def fake_post(url, timeout, json=None, headers=None):
        requests.append((url, json, headers, timeout))
        payload = (
            {"reservation": "one-use-reservation"}
            if url.endswith("/reservations")
            else {"id": "r-12345678"}
        )
        return httpx.Response(202, json=payload, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    private = LoopbackPrivateClient("http://127.0.0.1:8000")
    reservation = private.reserve()
    assert reservation == "one-use-reservation"
    assert private.start("public question", "baseline", "fast", 90, reservation) == "r-12345678"
    with pytest.raises(ValueError):
        private.get("../../settings")
    assert len(requests) == 2
    assert requests[0][0] == "http://127.0.0.1:8000/internal/public/reservations"
    assert requests[1][0] == "http://127.0.0.1:8000/runs"
    assert requests[1][1]["origin"] == "public"
    assert requests[1][1]["deadline_seconds"] == 90
    assert requests[1][2] == {"X-Public-Reservation": reservation}


def test_reservation_race_rejects_without_starting_private_run():
    import threading

    from harness_public_gateway.app import Admission

    class ReservationLost(FakePrivate):
        def reserve(self):
            raise RuntimeError("private capacity race detail")

    private = ReservationLost()
    admission = Admission(private)
    entry = admission.enqueue("Question is sufficiently long", "fast", "client")
    for _ in range(50):
        if entry.state == "rejected":
            break
        threading.Event().wait(0.02)
    assert entry.state == "rejected"
    assert entry.failure == "service busy"
    assert private.calls == []
