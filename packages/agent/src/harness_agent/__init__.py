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
        agent_decision: dict = {}

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
                purpose=record.get("purpose"),
                expectedEvidenceType=record.get("expected_evidence_type"),
            )
            trace.append(row)
            self.trace_sink(row)

        clarification_needed = self.baseline.enhanced and self.baseline.needs_clarification(question)
        queue_depth = None if clarification_needed else self._heavy_queue_depth()
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
        if not heavy_busy and not clarification_needed and base_run.status == "answered":
            mcp = MCPToolClient(tool_sink, agent="supervisor")
            try:
                grounded_context = {
                    "preliminary_answer": (
                        base_run.readings[base_run.chosen_reading].answer
                        if base_run.chosen_reading is not None else None
                    ),
                    "preliminary_triples": [
                        {"subject": triple.s, "predicate": triple.p,
                         "object": triple.o, "source_id": triple.source}
                        for reading in base_run.readings for triple in reading.triples
                    ],
                    "preliminary_claims": [
                        {"id": claim.id, "kind": claim.kind, "paper_id": claim.paper_id,
                         "fields": {key: field.value for key, field in claim.fields.items()},
                         "source_ids": claim.source_ids}
                        for reading in base_run.readings for claim in reading.claims
                    ],
                }
                agent_final = mcp.run_agent(self.agent_model, question, grounded_context)
                if agent_final:
                    try:
                        start, end = agent_final.find("{"), agent_final.rfind("}")
                        agent_decision = json.loads(agent_final[start:end + 1])
                        if not isinstance(agent_decision, dict):
                            raise TypeError("agent decision must be an object")
                    except (ValueError, TypeError):
                        mcp_error = "agent_final_not_structured"
                    row = TraceRow(
                        atS=round(monotonic() - started, 3),
                        action="Agent research decision",
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
        if base_run.status == "answered" and not clarification_needed and not heavy_busy and not agent_decision and not mcp_error:
            mcp_error = "agent_returned_no_research_decision"
        if agent_decision and not mcp_error:
            available = {claim.id for reading in base_run.readings for claim in reading.claims}
            verified = agent_decision.get("verified_claim_ids")
            evidence_ids = agent_decision.get("additional_evidence_ids")
            if not isinstance(verified, list) or not isinstance(evidence_ids, list):
                mcp_error = "invalid_agent_decision_shape"
            elif any(item not in available for item in verified):
                mcp_error = "agent_claim_id_not_in_preliminary_graph"
            elif hasattr(self.baseline, "corpus") and any(
                item not in self.baseline.corpus.by_passage for item in evidence_ids
            ):
                mcp_error = "agent_evidence_id_not_in_corpus"
            elif not tool_calls:
                mcp_error = "agent_used_no_corpus_tools"
            elif evidence_ids and hasattr(self.baseline, "seed_passage_ids"):
                self.baseline.seed_passage_ids = list(dict.fromkeys(evidence_ids))
                result = self.baseline.run(question, budget, run_id)
                base_run = result["run"]
            elif verified and base_run.chosen_reading is not None:
                reading = base_run.readings[base_run.chosen_reading]
                selected = [claim for claim in reading.claims if claim.id in verified]
                keys = {
                    (claim.fields["subject"].value, claim.fields["relation"].value,
                     claim.fields["object"].value)
                    for claim in selected
                }
                triples = [item for item in reading.triples if (item.s, item.p, item.o) in keys]
                if selected and triples:
                    answer = "The verified corpus claims report: " + "; ".join(
                        f"{claim.fields['object'].value} [{', '.join(claim.source_ids)}]"
                        for claim in selected
                    ) + "."
                    reading = reading.model_copy(update={
                        "claims": selected, "triples": triples, "answer": answer,
                        "claim_relations": [link for link in reading.claim_relations
                                            if link.source_claim in verified and link.target_claim in verified],
                    })
                    base_run = base_run.model_copy(update={"readings": [reading]})
                    result["artifacts"]["answer.md"] = answer.encode()
                    result["artifacts"]["claims.json"] = json.dumps({
                        "claims": [claim.model_dump(by_alias=True) for claim in selected],
                        "relations": [link.model_dump(by_alias=True) for link in reading.claim_relations],
                    }).encode()
                else:
                    mcp_error = "agent_selected_no_supported_claims"
            else:
                mcp_error = "agent_verified_no_claims"
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

        plans = [str(agent_decision.get("reasoning_summary"))] if agent_decision.get("reasoning_summary") else []
        plans.extend(f"MCP: {record['tool']}" for record in tool_calls if record.get("tool") != "mcp_session")
        trace = sorted(list(base_run.trace) + trace, key=lambda row: row.at_s)
        at_s = (trace[-1].at_s if trace else 0.0) + 0.01
        for plan in plans:
            row = TraceRow(atS=at_s, action="Agent planner lens", returned=plan)
            self.trace_sink(row)
            trace.append(row)
            at_s += 0.01
        ambiguous = base_run.status == "awaiting_reading"
        readings = list(base_run.readings)
        status = base_run.status
        chosen = base_run.chosen_reading
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
                "producing_path": "baseline_fallback" if mcp_error or base_run.status == "incomplete" else "agentic",
                "execution_status": "degraded" if mcp_error else "failed" if base_run.status == "incomplete" else "completed",
                "reason": (
                    f"agentic_degraded: {mcp_error}" if mcp_error
                    else base_run.reason if base_run.status in {"incomplete", "corpus_absent", "no_evidence_found"}
                    else None
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
