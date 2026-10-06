import json

import httpx
import pytest
from harness_extraction import (
    ModelFailure,
    OpenAICompatibleModel,
    model_events,
    model_purpose,
)


def use_transport(monkeypatch, handler) -> list[dict]:
    """Route the client's HTTP calls to ``handler``; return the request bodies seen."""
    seen: list[dict] = []
    real = httpx.Client

    def recording(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return handler(request)

    monkeypatch.setattr("harness_extraction.httpx.Client",
                        lambda timeout: real(transport=httpx.MockTransport(recording), timeout=timeout))
    return seen


def reply(content: str, finish: str = "stop", tokens: int = 12, **message) -> httpx.Response:
    return httpx.Response(200, json={
        "choices": [{"message": {"content": content, **message}, "finish_reason": finish}],
        "usage": {"completion_tokens": tokens},
    })


def test_reasoning_is_ignored_and_the_call_is_visible(monkeypatch) -> None:
    seen = use_transport(monkeypatch, lambda _: reply('<think>maybe {"x": 1}?</think>{"ok": true}', tokens=640))
    monkeypatch.setenv("HARNESS_MODEL_OPTIONS", json.dumps(
        {"heavy-model": {"max_tokens": 6144, "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}}))
    events: list[tuple[str, str]] = []
    with model_events(lambda action, detail: events.append((action, detail))), model_purpose("schema inference"):
        value = OpenAICompatibleModel(model="heavy-model").complete_json("s", "u")
    assert value == {"ok": True}
    assert seen[0]["max_tokens"] == 6144
    assert seen[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert [action for action, _ in events] == ["Model call started", "Model call finished"]
    assert "heavy-model · schema inference" in events[1][1] and "640 tokens" in events[1][1]


def test_output_limit_is_reported_and_not_retried(monkeypatch) -> None:
    seen = use_transport(monkeypatch, lambda _: reply("<think>long reasoning", finish="length"))
    with pytest.raises(ModelFailure, match="output limit"):
        OpenAICompatibleModel(model="heavy-model").complete_json("s", "u")
    assert len(seen) == 1


def test_timeouts_are_not_retried(monkeypatch) -> None:
    def slow(_: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")
    seen = use_transport(monkeypatch, slow)
    events: list[tuple[str, str]] = []
    with (
        model_events(lambda action, detail: events.append((action, detail))),
        pytest.raises(ModelFailure, match="timed out"),
    ):
        OpenAICompatibleModel(model="heavy-model").complete_json("s", "u")
    assert len(seen) == 1
    assert events[-1][0] == "Model call failed"


def test_transient_server_errors_are_retried(monkeypatch) -> None:
    answers = iter([httpx.Response(503, text="overloaded"), reply('{"ok": 1}')])
    seen = use_transport(monkeypatch, lambda _: next(answers))
    assert OpenAICompatibleModel(model="m").complete_json("s", "u") == {"ok": 1}
    assert len(seen) == 2


def test_client_errors_surface_the_provider_message(monkeypatch) -> None:
    use_transport(monkeypatch, lambda _: httpx.Response(400, text='{"error": "context too long"}'))
    with pytest.raises(ModelFailure, match="HTTP 400.*context too long"):
        OpenAICompatibleModel(model="m").complete_json("s", "u")


def test_streaming_reply_is_assembled_like_a_plain_reply(monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_MODEL_STREAM", "true")
    chunks = [
        {"choices": [{"delta": {"content": '{"ok": '}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "true}"}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"completion_tokens": 5}},
    ]
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
    seen = use_transport(monkeypatch, lambda _: httpx.Response(
        200, text=body, headers={"content-type": "text/event-stream"}))
    events: list[tuple[str, str]] = []
    with model_events(lambda action, detail: events.append((action, detail))):
        assert OpenAICompatibleModel(model="m").complete_json("s", "u") == {"ok": True}
    assert seen[0]["stream"] is True and seen[0]["stream_options"] == {"include_usage": True}
    assert "5 tokens" in events[-1][1]


def test_prompt_template_is_named_in_the_finished_event(monkeypatch) -> None:
    from harness_extraction import complete
    from harness_prompts import render

    use_transport(monkeypatch, lambda _: reply('{"definitions": []}'))
    events: list[tuple[str, str]] = []
    prompt = render("extraction/edc_define", {"relations": ["uses"]}, prompt_set="v2")
    with model_events(lambda action, detail: events.append((action, detail))):
        complete(OpenAICompatibleModel(model="m"), prompt)
    assert f"prompt {prompt.ref}" in events[-1][1]


def test_the_reply_budget_never_exceeds_what_the_context_window_leaves(monkeypatch) -> None:
    from harness_extraction import ContextOverflow

    seen = use_transport(monkeypatch, lambda _: reply('{"ok": true}'))
    monkeypatch.setenv("HARNESS_MODEL_CONTEXT_TOKENS", "8192")
    OpenAICompatibleModel(model="heavy-model").complete_json("s", "x" * 15000)        # about 5,000 tokens
    assert seen[-1]["max_tokens"] < 4096 and seen[-1]["max_tokens"] > 2500
    with pytest.raises(ContextOverflow, match="context_window_exceeded"):
        OpenAICompatibleModel(model="heavy-model").complete_json("s", "x" * 30000)
    assert len(seen) == 1                                                             # refused before sending
    monkeypatch.setenv("HARNESS_MODEL_OPTIONS", json.dumps({"heavy-model": {"context_tokens": 65536}}))
    OpenAICompatibleModel(model="heavy-model").complete_json("s", "x" * 30000)
    assert seen[-1]["max_tokens"] == 4096


def test_a_provider_context_error_is_named_as_such(monkeypatch) -> None:
    from harness_extraction import ContextOverflow

    use_transport(monkeypatch, lambda _: httpx.Response(400, text='{"error": "litellm.ContextWindowExceededError: '
                                                                   'maximum context length is 32768 tokens"}'))
    with pytest.raises(ContextOverflow, match="context_window_exceeded"):
        OpenAICompatibleModel(model="heavy-model").complete_json("s", "u")
