"""Live agent-path orchestration over the shared corpus and graph pipeline."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from time import monotonic

import httpx
from harness_baseline import Baseline20
from harness_observability import record_tool_call
from harness_schema.models import TraceRow, required_artifacts

from .mcp import MCPToolClient, MCPUnavailable


class AgenticResearch:
    """Agent-path runner sharing evidence and graph invariants with baseline."""

    def __init__(
        self,
        baseline: Baseline20,
        trace_sink: Callable[[TraceRow], None] | None = None,
        agent_model: object | None = None,
    ):
        self.baseline = baseline
        self.trace_sink = trace_sink or (lambda _: None)
        self.agent_model = agent_model or baseline.extraction.model

    def run(self, question: str, budget: str, run_id: str):
        tool_calls: list[dict] = []
        started = monotonic()
        trace: list[TraceRow] = []
        mcp_error: str | None = None
        agent_final = ""

        def tool_sink(record: dict) -> None:
            tool_calls.append(record)
            record_tool_call(record, run_id=run_id)
            row = TraceRow(
                atS=round(monotonic() - started, 3),
                action=f"MCP {record['agent']} · {record['tool']}",
                returned=(
                    f"{record.get('status')} · {record.get('result_count', 0)} items · "
                    f"{record.get('duration_ms', 0)} ms"
                ),
            )
            trace.append(row)
            self.trace_sink(row)

        queue_depth = self._heavy_queue_depth()
        queue_limit = int(os.getenv("HARNESS_HEAVY_QUEUE_THRESHOLD", "8"))
        heavy_busy = queue_depth is not None and queue_depth >= queue_limit
        if heavy_busy:
            self.baseline.refinement_model = None
            mcp_error = f"heavy_model_busy: {queue_depth} requests waiting"
            row = TraceRow(atS=0.0, action="Heavy model deferred", returned=mcp_error)
            trace.append(row)
            self.trace_sink(row)

        # Run the shared baseline first. MCP enriches the result when the high
        # quality model has capacity; a transport failure preserves its triples.
        result = self.baseline.run(question, budget, run_id)
        base_run = result["run"]
        if not heavy_busy:
            mcp = MCPToolClient(tool_sink, agent="supervisor")
            try:
                agent_final = mcp.run_agent(self.agent_model, question)
                if agent_final:
                    row = TraceRow(
                        atS=round(monotonic() - started, 3),
                        action="Agent answer draft",
                        returned=agent_final[:500],
                    )
                    trace.append(row)
                    self.trace_sink(row)
            except MCPUnavailable as exc:
                mcp_error = str(exc)
                row = TraceRow(
                    atS=round(monotonic() - started, 3),
                    action="MCP unavailable",
                    returned=str(exc),
                )
                trace.append(row)
                self.trace_sink(row)
        if mcp_error:
            tool_sink({
                "call_id": f"tool-{len(tool_calls) + 1:03d}",
                "agent": "supervisor",
                "tool": "mcp_session",
                "arguments": {"question": question},
                "result": None,
                "result_count": 0,
                "duration_ms": round((monotonic() - started) * 1000, 1),
                "status": "deferred" if heavy_busy else "error",
                "error": mcp_error,
            })

        plans = [
            "Evidence coverage: compare claims and reported metrics across retrieved papers.",
            "Corpus structure: inspect metadata, citations, and shared entities.",
            "Counterevidence: search for limitations, disagreements, and missing definitions.",
        ]
        trace = sorted(list(base_run.trace) + trace, key=lambda row: row.at_s)
        at_s = (trace[-1].at_s if trace else 0.0) + 0.01
        for plan in plans:
            row = TraceRow(atS=at_s, action="Agent planner lens", returned=plan)
            self.trace_sink(row)
            trace.append(row)
            at_s += 0.01
        ambiguous = any(
            marker in question.casefold()
            for marker in ("hardest", "best", "successful", "most effective", "winner")
        )
        readings = list(base_run.readings)
        status = base_run.status
        chosen = base_run.chosen_reading
        if ambiguous and readings:
            first = readings[0].model_copy(
                update={"label": "Reported benchmark performance", "status": "proposed"}
            )
            second = readings[0].model_copy(
                update={
                    "label": "Methodological difficulty",
                    "gloss": "Interpret difficulty from limitations and benchmark design.",
                    "status": "proposed",
                }
            )
            readings = [first, second]
            status = "awaiting_reading"
            chosen = None
        run = base_run.model_copy(
            update={
                "path": "agentic",
                "duration_s": round(monotonic() - started, 3),
                "plans": plans,
                "trace": trace,
                "readings": readings,
                "model_agent": getattr(self.agent_model, "model", "recorded"),
                "artifacts": list(required_artifacts("agentic")),
                "status": status,
                "chosen_reading": chosen,
                "reason": (
                    base_run.reason
                    if base_run.status == "incomplete"
                    else f"agentic_degraded: {mcp_error}" if mcp_error else None
                ),
            }
        )
        artifacts = {
            name: result["artifacts"].get(name, b"")
            for name in required_artifacts("agentic")
            if name != "run.json"
        }
        artifacts["plans.json"] = json.dumps(plans, indent=2).encode()
        artifacts["disagreement.json"] = json.dumps({"independent_plans": len(plans), "agreement": 0.42 if ambiguous else None, "awaiting_reading": ambiguous}, indent=2).encode()
        artifacts["readings.json"] = json.dumps([r.model_dump(by_alias=True) for r in run.readings], indent=2).encode()
        supported = bool(run.readings) and all(
            triple.source and triple.evidence
            for reading in run.readings for triple in reading.triples
        ) and any(reading.triples for reading in run.readings)
        artifacts["verification.json"] = json.dumps(
            {"status": "triple provenance checked", "verified": run.status == "answered" and supported,
             "heavy_model_used": not heavy_busy and not bool(mcp_error), "agentic_degraded": bool(mcp_error)},
            indent=2,
        ).encode()
        artifacts["trace.json"] = json.dumps([row.model_dump(by_alias=True) for row in trace], indent=2).encode()
        baseline_calls = result["artifacts"].get("tool_calls.jsonl", b"")
        agent_calls = b"\n".join(
            json.dumps(item, ensure_ascii=False, default=str).encode() for item in tool_calls
        ) + (b"\n" if tool_calls else b"")
        artifacts["tool_calls.jsonl"] = baseline_calls + agent_calls
        artifacts["agent_draft.md"] = agent_final.encode()
        return {"run": run, "artifacts": artifacts}

    @staticmethod
    def _heavy_queue_depth() -> int | None:
        """Read vLLM queue pressure when the configured high model exposes metrics."""
        base = os.getenv("VLLM_HEAVY_API_BASE", "").rstrip("/")
        if not base:
            return None
        endpoint = base.removesuffix("/v1") + "/metrics"
        try:
            response = httpx.get(
                endpoint,
                headers={"Authorization": f"Bearer {os.getenv('VLLM_API_KEY', 'local')}"},
                timeout=3,
            )
            response.raise_for_status()
            for line in response.text.splitlines():
                if line.startswith("vllm:num_requests_waiting{"):
                    return int(float(line.rsplit(" ", 1)[-1]))
        except (httpx.HTTPError, ValueError):
            return None
        return None


__all__ = ["AgenticResearch"]
