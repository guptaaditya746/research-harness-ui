import json
from datetime import UTC, datetime

from harness_agent import AgenticResearch
from harness_agent.mcp import MCPToolClient, _tool_payload
from harness_schema.models import Metrics, Reading, Run, Triple, required_artifacts


class StubBaseline:
    enhanced = True
    refinement_model = object()

    def run(self, question: str, budget: str, run_id: str, concept: dict | None = None) -> dict:
        self.concept = concept
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


CONCEPT = {"label": "Introducing paper", "gloss": "Find the paper that introduced the method.",
           "classes": ["Paper"]}


class ConceptModel:
    model = "granite"

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = 0

    def complete_json(self, _system: str, _user: str) -> dict:
        self.calls += 1
        return self.payloads.pop(0)


class SearchCorpus:
    def search(self, query: str, limit: int = 10) -> dict:
        hits = 0 if "quantum" in query else 4
        return {"total": hits, "items": [{"passage_id": "p4-abstract"}] if hits else []}


def concept(label: str, term: str = "video prediction") -> dict:
    return {"label": label, "gloss": f"Research {label.lower()}.", "classes": ["Paper", "Bogus"],
            "search_terms": [term]}


def test_agent_always_proposes_three_to_four_concepts_before_research() -> None:
    baseline = StubBaseline()
    baseline.corpus = SearchCorpus()
    model = ConceptModel([{"concepts": [
        concept("Quantum angle", "quantum"), concept("Introducing paper"),
        concept("Benchmark results"), concept("Citation influence"),
    ]}])
    result = AgenticResearch(baseline, agent_model=model).run(
        "Which paper introduced Video Prediction Policy?", "slow", "r-concepts"
    )
    run = result["run"]
    assert run.status == "awaiting_reading"
    assert run.path == "agentic"
    assert run.chosen_reading is None
    assert [r.label for r in run.readings] == [
        "Introducing paper", "Benchmark results", "Citation influence", "Quantum angle"
    ]
    assert all(r.classes == ["Paper"] for r in run.readings)
    plans = json.loads(result["artifacts"]["plans.json"])
    assert plans[0]["evidence_hits"] == 4 and plans[-1]["evidence_hits"] == 0
    assert set(result["artifacts"]) == set(required_artifacts("agentic")) - {"run.json"}


def test_too_few_concepts_retries_then_fails_visibly() -> None:
    model = ConceptModel([{"concepts": [concept("Only one")]}, {"concepts": []}])
    run = AgenticResearch(StubBaseline(), agent_model=model).run("Q?", "fast", "r-few")["run"]
    assert model.calls == 2
    assert run.status == "incomplete"
    assert run.reason.startswith("concept_proposal_failed")


def test_busy_high_model_preserves_triples_and_marks_agentic_result_degraded(monkeypatch) -> None:
    monkeypatch.setattr(AgenticResearch, "_heavy_queue_depth", staticmethod(lambda: 94))
    baseline = StubBaseline()
    result = AgenticResearch(baseline, agent_model=HighModel()).run(
        "Which paper introduced Video Prediction Policy?", "slow", "r-test", concept=CONCEPT
    )
    run = result["run"]

    assert run.status == "answered"
    assert run.reason == "agentic_degraded: heavy_model_busy: 94 requests waiting"
    assert run.readings[0].triples[0].o == "Video Prediction Policy"
    assert baseline.refinement_model is None
    assert json.loads(result["artifacts"]["verification.json"])["agentic_degraded"] is True
    assert b'"status": "deferred"' in result["artifacts"]["tool_calls.jsonl"]


def test_agent_receives_preliminary_grounded_context(monkeypatch) -> None:
    monkeypatch.setattr(AgenticResearch, "_heavy_queue_depth", staticmethod(lambda: 0))
    captured = {}

    def run_agent(self, model, question, grounded_context):
        captured.update(grounded_context)
        return "Verified Video Prediction Policy [p4-metadata]."

    monkeypatch.setattr(MCPToolClient, "run_agent", run_agent)
    result = AgenticResearch(StubBaseline(), agent_model=HighModel()).run(
        "Which paper introduced Video Prediction Policy?", "slow", "r-context", concept=CONCEPT
    )

    assert result["run"].status == "answered"
    assert captured["preliminary_triples"] == [
        {"subject": "p4", "predicate": "title", "object": "Video Prediction Policy", "source_id": "p4-metadata"}
    ]
    assert "Video Prediction Policy" in captured["preliminary_answer"]


def test_mcp_result_text_is_unwrapped_for_model_context() -> None:
    result = {"content": [{"type": "text", "text": '{"total": 1, "items": [{"fact_id": "f1"}]}'}]}

    assert _tool_payload(result) == {"total": 1, "items": [{"fact_id": "f1"}]}
