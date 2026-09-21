"""One process-wide Langtrace setup and a smoke span."""

import os
from threading import Lock

from opentelemetry import trace

_lock = Lock()
_ready = False


def init() -> bool:
    """Return whether remote export is configured. Safe to call repeatedly."""
    global _ready
    with _lock:
        if _ready:
            return True
        os.environ.pop("LANGSMITH_TRACING", None)
        os.environ.pop("LANGCHAIN_TRACING_V2", None)
        key = os.getenv("LANGTRACE_API_KEY", "").strip()
        host = os.getenv("LANGTRACE_API_HOST", "").strip()
        if not (key or host):
            return False
        from langtrace_python_sdk import langtrace

        kwargs: dict[str, str | bool] = {
            "batch": False,
            "service_name": os.getenv("OTEL_SERVICE_NAME", "research-harness"),
        }
        if key:
            kwargs["api_key"] = key
        if host:
            kwargs["api_host"] = host
        langtrace.init(**kwargs)
        _ready = True
        return True


def hello_span() -> str:
    with trace.get_tracer("research-harness").start_as_current_span("harness.m0.hello") as span:
        span.set_attribute("harness.stage", "m0.hello")
        context = span.get_span_context()
        return f"{context.trace_id:032x}"
