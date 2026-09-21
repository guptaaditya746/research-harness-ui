"""One process-wide Langfuse setup and a smoke span."""

import os
from threading import Lock

from opentelemetry import trace

_lock = Lock()
_ready = False
_client = None


def init() -> bool:
    """Return whether remote export is configured. Safe to call repeatedly."""
    global _ready, _client
    with _lock:
        if _ready:
            return True
        os.environ.pop("LANGSMITH_TRACING", None)
        os.environ.pop("LANGCHAIN_TRACING_V2", None)
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
        secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
        base_url = os.getenv("LANGFUSE_BASE_URL", "").strip()
        if not (public_key and secret_key):
            return False
        from langfuse import Langfuse

        _client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            base_url=base_url or None,
            environment=os.getenv("HARNESS_ENV", "development"),
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
