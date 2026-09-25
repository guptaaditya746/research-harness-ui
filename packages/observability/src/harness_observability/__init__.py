"""One process-wide Langfuse setup and a smoke span."""

import logging
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from threading import Lock
from typing import Any

from harness_config import settings
from opentelemetry import trace

_lock = Lock()
_ready = False
_client = None
_logger = logging.getLogger(__name__)


def init() -> bool:
    """Return whether remote export is configured. Safe to call repeatedly."""
    global _ready, _client
    with _lock:
        if _ready:
            return True
        os.environ.pop("LANGSMITH_TRACING", None)
        os.environ.pop("LANGCHAIN_TRACING_V2", None)
        config = settings()
        public_key = config.langfuse_public_key.strip()
        secret_key = config.langfuse_secret_key.strip()
        base_url = config.langfuse_base_url.strip()
        if not (public_key and secret_key):
            return False
        from langfuse import Langfuse

        _client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            base_url=base_url or None,
            environment=config.harness_env,
            additional_headers={"x-langfuse-ingestion-version": "4"},
        )
        _ready = True
        return True


def hello_span() -> str:
    if _client is not None:
        with _client.start_as_current_observation(
            as_type="span", name="harness.m0.hello", metadata={"harness.stage": "m0.hello"}
        ) as observation:
            trace_id = observation.trace_id
        _client.flush()
        return trace_id
    with trace.get_tracer("research-harness").start_as_current_span("harness.m0.hello") as span:
        span.set_attribute("harness.stage", "m0.hello")
        context = span.get_span_context()
        return f"{context.trace_id:032x}"


def _bounded(value: Any, limit: int = 2000) -> Any:
    if isinstance(value, dict):
        return {str(k): _bounded(v, limit) for k, v in list(value.items())[:50]}
    if isinstance(value, (list, tuple)):
        return [_bounded(v, limit) for v in list(value)[:50]]
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "…"


@contextmanager
def run_observation(run_id: str, question: str, path: str, budget: str) -> Iterator[Any]:
    """Create a Langfuse parent observation without making telemetry mandatory."""
    if _client is None:
        yield None
        return
    try:
        observation = _client.start_as_current_observation(
            as_type="span",
            name="research.run",
            input={"run_id": run_id, "question": _bounded(question), "path": path, "budget": budget},
            metadata={"harness.run_id": run_id, "harness.path": path},
        )
    except Exception:  # noqa: BLE001
        yield None
        return
    try:
        with observation as active:
            yield active
    finally:
        # Export in the background: an unreachable collector must not delay
        # persisting the run the user is waiting for.
        threading.Thread(target=_flush_quietly, name="langfuse-flush", daemon=True).start()


def _flush_quietly() -> None:
    try:
        if _client is not None:
            _client.flush()
    except Exception:
        _logger.debug("Langfuse flush failed", exc_info=True)


def record_stage(name: str, returned: str, *, run_id: str | None = None) -> None:
    """Record a short child observation for a visible pipeline stage."""
    if _client is None:
        return
    try:
        with _client.start_as_current_observation(
            as_type="span",
            name=f"research.{name.lower().replace(' ', '_')}",
            output={"returned": _bounded(returned)},
            metadata={"harness.run_id": run_id} if run_id else None,
        ):
            pass
    except Exception:
        _logger.debug("Langfuse stage observation failed", exc_info=True)


def record_tool_call(record: dict[str, Any], *, run_id: str | None = None) -> None:
    """Record an MCP call while keeping large payloads out of telemetry."""
    if _client is None:
        return
    try:
        tool = str(record.get("tool", "unknown"))
        output = {
            "status": record.get("status"),
            "result_count": record.get("result_count"),
            "duration_ms": record.get("duration_ms"),
            "error": record.get("error"),
            "result_preview": _bounded(record.get("result")),
        }
        with _client.start_as_current_observation(
            as_type="span",
            name=f"research.tool_call.{tool}",
            input={"arguments": _bounded(record.get("arguments", {}))},
            output=output,
            metadata={
                "harness.run_id": run_id,
                "harness.agent": record.get("agent", "supervisor"),
                "harness.call_id": record.get("call_id"),
            },
        ):
            pass
    except Exception:
        _logger.debug("Langfuse tool observation failed", exc_info=True)
