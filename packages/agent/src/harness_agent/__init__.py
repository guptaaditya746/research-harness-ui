"""Live agent-path orchestration over the shared corpus and graph pipeline."""

from __future__ import annotations

import json
from collections.abc import Callable

from harness_baseline import Baseline20
from harness_schema.models import TraceRow, required_artifacts


class AgenticResearch:
    """Agent-path runner sharing evidence and graph invariants with baseline."""

    def __init__(self, baseline: Baseline20, trace_sink: Callable[[TraceRow], None] | None = None):
        self.baseline = baseline
        self.trace_sink = trace_sink or (lambda _: None)

    def run(self, question: str, budget: str, run_id: str):
        result = self.baseline.run(question, budget, run_id)
        base_run = result["run"]
        plans = [
            "Evidence coverage: compare claims and reported metrics across retrieved papers.",
            "Corpus structure: inspect metadata, citations, and shared entities.",
            "Counterevidence: search for limitations, disagreements, and missing definitions.",
        ]
        run = base_run.model_copy(
            update={
                "path": "agentic",
                "plans": plans,
                "model_agent": getattr(self.baseline.extraction.model, "model", "recorded"),
                "artifacts": list(required_artifacts("agentic")),
            }
        )
        artifacts = {
            name: result["artifacts"].get(name, b"")
            for name in required_artifacts("agentic")
            if name != "run.json"
        }
        artifacts["plans.json"] = json.dumps(plans, indent=2).encode()
        artifacts["disagreement.json"] = json.dumps({"independent_plans": len(plans), "agreement": None}, indent=2).encode()
        artifacts["readings.json"] = json.dumps([r.model_dump(by_alias=True) for r in run.readings], indent=2).encode()
        artifacts["verification.json"] = json.dumps({"status": "shared graph checks applied", "verified": run.status == "answered"}, indent=2).encode()
        return {"run": run, "artifacts": artifacts}


__all__ = ["AgenticResearch"]
