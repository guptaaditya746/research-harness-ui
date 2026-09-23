import json
from datetime import UTC, datetime

from harness_agent import AgenticResearch
from harness_schema.models import Metrics, Reading, Run, Triple, required_artifacts


class StubBaseline:
    refinement_model = object()

    def run(self, question: str, budget: str, run_id: str) -> dict:
        reading = Reading(
            label="Corpus facts", gloss="Verified facts", classes=["Paper"], axioms=[],
            sparql="", answer="Video Prediction Policy [p4-metadata].",
            triples=[Triple(s="p4", p="title", o="Video Prediction Policy", support="1 fact", evidence="Video Prediction Policy", source="p4-metadata")],
            metrics=Metrics(), status="answered",
        )
        run = Run(
            id=run_id, startedAt=datetime.now(UTC), durationS=0.01,
            question=question, path="baseline-enhanced", budget=budget, status="answered",
            readings=[reading], chosenReading=0, artifacts=list(required_artifacts("baseline-enhanced")),
        )
        return {"run": run, "artifacts": {"tool_calls.jsonl": b""}}


class HighModel:
    model = "heavy-model"


def test_busy_high_model_preserves_triples_and_marks_agentic_result_degraded(monkeypatch) -> None:
    monkeypatch.setattr(AgenticResearch, "_heavy_queue_depth", staticmethod(lambda: 94))
    baseline = StubBaseline()
    result = AgenticResearch(baseline, agent_model=HighModel()).run(
        "Which paper introduced Video Prediction Policy?", "slow", "r-test"
    )
    run = result["run"]

    assert run.status == "answered"
    assert run.reason == "agentic_degraded: heavy_model_busy: 94 requests waiting"
    assert run.readings[0].triples[0].o == "Video Prediction Policy"
    assert baseline.refinement_model is None
    assert json.loads(result["artifacts"]["verification.json"])["agentic_degraded"] is True
    assert b'"status": "deferred"' in result["artifacts"]["tool_calls.jsonl"]
