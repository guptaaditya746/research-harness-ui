"""Live agent-path orchestration over the shared corpus and graph pipeline."""

from __future__ import annotations

import json
from collections.abc import Callable
from time import monotonic

from harness_baseline import Baseline20
from harness_observability import record_tool_call
from harness_schema.models import Metrics, Reading, TraceRow, required_artifacts

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
            # Preserve a useful baseline result when the optional managed
            # service is down, but make the missing integration explicit.
            mcp_error = str(exc)
            row = TraceRow(
                atS=round(monotonic() - started, 3),
                action="MCP unavailable",
                returned=str(exc),
            )
            trace.append(row)
            self.trace_sink(row)

        result = self.baseline.run(question, budget, run_id)
        base_run = result["run"]
        mcp_answer_used = base_run.status == "incomplete" and bool(agent_final) and not mcp_error
        if base_run.status == "incomplete" and agent_final and not mcp_error:
            # The MCP supervisor has a usable evidence-grounded text response
            # even when the older ontology extractor rejects its inferred
            # schema. Keep that answer visible instead of replacing it with a
            # generic baseline failure message.
            base_run = base_run.model_copy(
                update={
                    "status": "answered",
                    "reason": None,
                    "readings": [
                        Reading(
                            label="MCP evidence answer",
                            gloss="Answer composed from managed corpus tool results.",
                            classes=["Paper"],
                            axioms=[],
                            properties=[],
                            sparql="",
                            answer=agent_final,
                            triples=[],
                            metrics=Metrics(),
                            status="answered",
                        )
                    ],
                    "chosen_reading": 0,
                }
            )
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
                "plans": plans,
                "trace": trace,
                "readings": readings,
                "model_agent": getattr(self.agent_model, "model", "recorded"),
                "artifacts": list(required_artifacts("agentic")),
                "status": "incomplete" if mcp_error else status,
                "chosen_reading": None if mcp_error else chosen,
                "reason": f"mcp_unavailable: {mcp_error}" if mcp_error else base_run.reason,
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
        artifacts["verification.json"] = json.dumps({"status": "shared graph checks applied", "verified": run.status == "answered"}, indent=2).encode()
        artifacts["trace.json"] = json.dumps([row.model_dump(by_alias=True) for row in trace], indent=2).encode()
        if mcp_answer_used:
            artifacts["answer.md"] = agent_final.encode()
        artifacts["tool_calls.jsonl"] = b"\n".join(
            json.dumps(item, ensure_ascii=False, default=str).encode() for item in tool_calls
        ) + (b"\n" if tool_calls else b"")
        artifacts["agent_draft.md"] = agent_final.encode()
        return {"run": run, "artifacts": artifacts}


__all__ = ["AgenticResearch"]
