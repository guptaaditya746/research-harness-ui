from harness_observability import hello_span
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


def test_hello_span_is_recorded(monkeypatch) -> None:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer", provider.get_tracer)

    trace_id = hello_span()
    spans = exporter.get_finished_spans()

    assert len(spans) == 1
    assert spans[0].name == "harness.m0.hello"
    assert spans[0].attributes["harness.stage"] == "m0.hello"
    assert trace_id == f"{spans[0].context.trace_id:032x}"
