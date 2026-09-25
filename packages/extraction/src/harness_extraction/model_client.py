"""The OpenAI-compatible client for structured (JSON) model calls.

All structured calls go through LiteLLM's /chat/completions. One call is
bounded in time and output, reports its progress as model events, and is
retried only for transient transport failures (connection errors, 429, 5xx).
A timeout is not retried: repeating the same long generation only multiplies
the wait. A reply cut off by the output limit is reported as such.
"""

from __future__ import annotations

import json
import re
import threading
import time
from typing import TYPE_CHECKING, Any, Protocol

import httpx
from harness_config import settings
from tenacity import (
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .call_control import bounded_timeout, run_interruptible
from .errors import ModelFailure
from .events import current_prompt, current_purpose, emit_model_event, using_prompt

if TYPE_CHECKING:
    from harness_prompts import Prompt


def model_options(alias: str) -> dict[str, Any]:
    """Per-alias request options from HARNESS_MODEL_OPTIONS (JSON), if configured.

    Example: {"heavy-model": {"max_tokens": 6144,
              "extra_body": {"chat_template_kwargs": {"enable_thinking": false}}}}
    """
    return settings().model_options(alias)


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Remove reasoning blocks a thinking model may emit before its answer."""
    text = _THINK_BLOCK.sub("", text)
    # An unterminated block means the answer never started.
    if re.search(r"<think>", text, re.IGNORECASE):
        text = re.split(r"<think>", text, flags=re.IGNORECASE)[0]
    return text.strip()


class ModelClient(Protocol):
    def complete_json(self, system: str, user: str) -> dict[str, Any]: ...


def complete(model: ModelClient, prompt: Prompt) -> dict[str, Any]:
    """Send a rendered prompt through ``model`` and record which template it was."""
    with using_prompt(prompt.ref):
        return model.complete_json(prompt.system, prompt.user)


def _json_object(text: str) -> dict[str, Any]:
    text = strip_reasoning(text)
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1) if fenced else text[text.find("{") : text.rfind("}") + 1]
    value = json.loads(candidate)
    if not isinstance(value, dict):
        raise TypeError("model response must be a JSON object")
    return value


class _ProviderStatus(Exception):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"HTTP {status}")
        self.status = status
        self.body = body


class _Transient(Exception):
    """A failed attempt worth retrying (connection error, 429, 5xx)."""

    def __init__(self, error: str) -> None:
        super().__init__(error)
        self.error = error


def _read_stream(response: httpx.Response, cancel: threading.Event, started: float,
                 timeout_s: float) -> dict[str, Any]:
    """Assemble a streamed chat completion into the non-streaming reply shape.

    Checks ``cancel`` between chunks: leaving the ``with client.stream`` block
    closes the connection, and the server aborts the generation.
    """
    content, reasoning, finish, usage = [], [], None, None
    for line in response.iter_lines():
        if cancel.is_set():
            raise ModelFailure("model call cancelled")
        if time.monotonic() - started > timeout_s:
            raise httpx.ReadTimeout("stream exceeded the call timeout")
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        usage = chunk.get("usage") or usage
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            content.append(delta.get("content") or "")
            reasoning.append(delta.get("reasoning_content") or "")
            finish = choice.get("finish_reason") or finish
    return {
        "choices": [{"message": {"content": "".join(content),
                                 "reasoning_content": "".join(reasoning) or None},
                     "finish_reason": finish}],
        "usage": usage or {},
    }


class OpenAICompatibleModel:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout_s: float = 30,
        retries: int | None = None,
    ) -> None:
        config = settings()
        self.base_url = (base_url or config.litellm_base_url).rstrip("/")
        self.model = model or config.model_small
        self.api_key = api_key or config.api_key
        self.timeout_s = timeout_s
        self.retries = retries or config.provider_max_retries

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        """One structured call, bounded in time and output, with visible progress."""
        config = settings()
        purpose = current_purpose()
        prompt = current_prompt()
        options = model_options(self.model)
        max_tokens = int(options.get("max_tokens") or config.max_output_tokens)
        body: dict[str, Any] = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **(options.get("extra_body") or {}),
        }
        if config.model_stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
        retrying = Retrying(
            stop=stop_after_attempt(self.retries),
            wait=wait_exponential(multiplier=0.25),
            retry=retry_if_exception_type(_Transient),
            reraise=False,
        )
        try:
            for attempt in retrying:
                with attempt:
                    return self._attempt(body, purpose, prompt, max_tokens,
                                         attempt.retry_state.attempt_number - 1)
        except RetryError as exc:
            last = exc.last_attempt.exception()
            error = last.error if isinstance(last, _Transient) else str(last)
            raise ModelFailure(f"provider failed after {self.retries} attempts: {error}") from None
        raise ModelFailure("no attempt made")  # pragma: no cover - Retrying always attempts once

    def _attempt(self, body: dict[str, Any], purpose: str, prompt: str | None, max_tokens: int,
                 attempt: int) -> dict[str, Any]:
        timeout_s = bounded_timeout(self.timeout_s)
        started = time.monotonic()
        emit_model_event(
            "Model call started",
            f"{self.model} · {purpose}" + (f" · attempt {attempt + 1}" if attempt else ""),
        )
        cancel = threading.Event()

        def request() -> dict[str, Any]:
            with httpx.Client(timeout=timeout_s) as client:
                url = f"{self.base_url}/chat/completions"
                headers = {"Authorization": f"Bearer {self.api_key}"}
                if body.get("stream"):
                    with client.stream("POST", url, headers=headers, json=body) as response:
                        if response.status_code >= 400:
                            response.read()
                            raise _ProviderStatus(response.status_code, response.text[:400])
                        return _read_stream(response, cancel, started, timeout_s)
                response = client.post(url, headers=headers, json=body)
                if response.status_code >= 400:
                    raise _ProviderStatus(response.status_code, response.text[:400])
                return response.json()

        try:
            payload = run_interruptible(request, cancel)
        except httpx.TimeoutException:
            took = time.monotonic() - started
            error = f"timed out after {took:.0f}s ({self.model} · {purpose})"
            emit_model_event("Model call failed", error)
            raise ModelFailure(error) from None
        except _ProviderStatus as exc:
            error = f"HTTP {exc.status}: {exc.body}"
            emit_model_event("Model call failed", f"{self.model} · {purpose} · {error}")
            if exc.status == 429 or exc.status >= 500:
                raise _Transient(error) from None
            raise ModelFailure(error) from None
        except httpx.HTTPError as exc:
            error = f"{type(exc).__name__}: {exc}"
            emit_model_event("Model call failed", f"{self.model} · {purpose} · {error}")
            raise _Transient(error) from None
        took = time.monotonic() - started
        try:
            choice = payload["choices"][0]
            content = str(choice["message"].get("content") or "")
            usage = payload.get("usage") or {}
            tokens = usage.get("completion_tokens")
            if choice.get("finish_reason") == "length":
                raise ValueError(
                    f"reply hit the {max_tokens}-token output limit before finishing"
                    + (" (a thinking model spent it on reasoning)"
                       if choice["message"].get("reasoning_content") or "<think>" in content else "")
                )
            value = _json_object(content)
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            error = f"unusable reply: {exc}"[:300]
            emit_model_event("Model call failed", f"{self.model} · {purpose} · {error} · {took:.1f}s")
            # A deterministic model repeats the same reply; do not retry.
            raise ModelFailure(error) from None
        emit_model_event(
            "Model call finished",
            f"{self.model} · {purpose} · {took:.1f}s" + (f" · {tokens} tokens" if tokens else "")
            + (f" · prompt {prompt}" if prompt else ""),
        )
        return value


__all__ = ["ModelClient", "OpenAICompatibleModel", "complete", "model_options", "strip_reasoning"]
