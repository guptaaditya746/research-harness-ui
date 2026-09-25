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


def test_agent_receives_selected_concept_before_baseline_answer(monkeypatch) -> None:
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
    assert captured["selected_concept"]["label"] == "Introducing paper"
    assert captured["preliminary_claims"] == []
    assert captured["preliminary_answer"] is None


def test_mcp_search_precedes_answer_verification_and_seeds_evidence(monkeypatch) -> None:
    monkeypatch.setattr(AgenticResearch, "_heavy_queue_depth", staticmethod(lambda: 0))
    order = []

    class VerifierSpy:
        calls = 0

        def complete_json(self, system, user):
            self.calls += 1
            return {"supported_ids": [], "unsupported_ids": [], "notes": "unused"}

    verifier = VerifierSpy()

    class RejectedBaseline(StubBaseline):
        def __init__(self):
            self.seed_passage_ids = []
            self.refinement_model = verifier

        def run(self, question, budget, run_id, concept=None):
            order.append(("baseline", list(self.seed_passage_ids)))
            result = super().run(question, budget, run_id, concept=concept)
            result["run"] = result["run"].model_copy(update={
                "status": "incomplete", "reason": "schema_rejected", "readings": [],
                "chosen_reading": None,
            })
            return result

    def run_agent(self, model, question, grounded_context):
        order.append(("mcp", grounded_context["preliminary_answer"]))
        self.sink({
            "call_id": "tool-001", "agent": "supervisor", "tool": "search_evidence",
            "arguments": {"query": "recurring themes"}, "status": "ok",
            "duration_ms": 3.0, "result_count": 1,
            "result": {"items": [{"passage_id": "p4-intro", "text": "Evidence."}]},
        })
        return json.dumps({
            "verified_claim_ids": [], "additional_evidence_ids": ["p4-intro"],
            "reasoning_summary": "found evidence",
        })

    monkeypatch.setattr(MCPToolClient, "run_agent", run_agent)
    result = AgenticResearch(RejectedBaseline(), agent_model=HighModel()).run(
        "What research themes recur?", "slow", "r-early", concept=CONCEPT,
    )

    assert order == [("mcp", None), ("baseline", ["p4-intro"])]
    assert result["run"].status == "incomplete"
    checks = {check.name: check for check in result["run"].checks}
    assert checks["agent research (MCP)"].status == "pass"
    assert checks["independent verification"].status == "skipped"
    assert verifier.calls == 0
    assert b'"passage_id": "p4-intro"' in result["artifacts"]["tool_calls.jsonl"]


def test_unstructured_agent_decision_still_supplies_observed_passages(monkeypatch) -> None:
    monkeypatch.setattr(AgenticResearch, "_heavy_queue_depth", staticmethod(lambda: 0))

    class SeededBaseline(StubBaseline):
        def __init__(self):
            self.seed_passage_ids = []
            self.seen_seeds = []

        def run(self, question, budget, run_id, concept=None):
            self.seen_seeds = list(self.seed_passage_ids)
            return super().run(question, budget, run_id, concept=concept)

    def run_agent(self, model, question, grounded_context):
        self.sink({
            "call_id": "tool-001", "agent": "supervisor", "tool": "search_evidence",
            "arguments": {"query": "NLP task"}, "status": "ok", "duration_ms": 3.0,
            "result_count": 2,
            "result": {"items": [
                {"passage_id": "p4-intro", "paper_id": "p4", "text": "Relevant task."},
                {"passage_id": "p4-method", "paper_id": "p4", "text": "Relevant approach."},
            ]},
        })
        return "The corpus includes several NLP tasks."

    monkeypatch.setattr(MCPToolClient, "run_agent", run_agent)
    baseline = SeededBaseline()
    result = AgenticResearch(baseline, agent_model=HighModel()).run(
        "Which NLP tasks appear most often?", "slow", "r-unstructured", concept=CONCEPT,
    )

    assert baseline.seen_seeds == ["p4-intro", "p4-method"]
    assert result["run"].execution_status == "degraded"
    assert result["run"].reason == "agentic_degraded: agent_final_not_structured"


def test_mcp_result_text_is_unwrapped_for_model_context() -> None:
    result = {"content": [{"type": "text", "text": '{"total": 1, "items": [{"fact_id": "f1"}]}'}]}

    assert _tool_payload(result) == {"total": 1, "items": [{"fact_id": "f1"}]}


def _agent_run(monkeypatch, evidence_ids: list[str]):
    """One agentic run whose agent searches once and returns the given evidence IDs."""
    monkeypatch.setattr(AgenticResearch, "_heavy_queue_depth", staticmethod(lambda: 0))

    def run_agent(self, model, question, grounded_context):
        self.sink({"call_id": "tool-001", "agent": "supervisor", "tool": "search_evidence",
                   "arguments": {"query": "video prediction"}, "status": "ok", "duration_ms": 3.0,
                   "result": {"total": 1, "items": [{"passage_id": "p4-intro", "text": "..."}]},
                   "result_count": 1})
        return json.dumps({"verified_claim_ids": [], "additional_evidence_ids": evidence_ids,
                           "reasoning_summary": "searched once"})

    monkeypatch.setattr(MCPToolClient, "run_agent", run_agent)
    baseline = StubBaseline()
    baseline.last_route = "verified_facts"
    result = AgenticResearch(baseline, agent_model=HighModel()).run(
        "Which paper introduced Video Prediction Policy?", "slow", "r-ids", concept=CONCEPT
    )
    return result["run"], json.loads(result["artifacts"]["verification.json"])


def test_agent_evidence_ids_must_have_been_observed_in_the_session(monkeypatch) -> None:
    run, verification = _agent_run(monkeypatch, ["p4-intro", "p99-invented"])
    checks = {check.name: check for check in run.checks}
    assert run.status == "answered" and run.execution_status != "degraded"
    assert checks["agent evidence IDs"].status == "warn"
    assert "p99-invented" in checks["agent evidence IDs"].detail
    assert checks["agent research (MCP)"].status == "pass"
    assert verification["agent_evidence_ids"] == {
        "supplied": ["p4-intro", "p99-invented"], "observed_in_session": ["p4-intro"],
        "rejected_unobserved": ["p99-invented"],
        "used_for_retrieval": ["p4-intro"], "used_for_verification": ["p4-intro"],
    }
    assert verification["triple_provenance"]["verified"] is True
    assert verification["agentic_execution"]["reason"] is None


def test_only_unobserved_evidence_ids_degrade_the_agentic_result(monkeypatch) -> None:
    run, verification = _agent_run(monkeypatch, ["p99-invented"])
    checks = {check.name: check for check in run.checks}
    assert checks["agent evidence IDs"].status == "fail"
    assert checks["agentic execution"].status == "warn"
    assert verification["agentic_execution"]["reason"].startswith("agent_evidence_ids_not_observed")
    # Provenance of the grounded answer is reported separately from agent execution.
    assert verification["triple_provenance"]["verified"] is True
    assert verification["agentic_degraded"] is True


class VerifierModel:
    model = "judge"

    def __init__(self, payload):
        self.payload = payload
        self.requests: list[dict] = []

    def complete_json(self, _system: str, user: str) -> dict:
        self.requests.append(json.loads(user))
        return self.payload


def test_independent_verifier_sees_only_cited_evidence_and_never_gates(monkeypatch) -> None:
    monkeypatch.setattr(AgenticResearch, "_heavy_queue_depth", staticmethod(lambda: 0))

    def run_agent(self, model, question, grounded_context):
        self.sink({"call_id": "tool-001", "agent": "supervisor", "tool": "search_evidence",
                   "arguments": {"query": "video prediction"}, "status": "ok", "duration_ms": 3.0,
                   "result": {"total": 2, "items": [
                       {"passage_id": "p4-intro", "text": "We introduce the Video Prediction Policy."},
                       {"passage_id": "p4-related", "text": "Prior work on diffusion policies."}]},
                   "result_count": 2})
        return json.dumps({"verified_claim_ids": [], "additional_evidence_ids": ["p4-intro", "p4-related"],
                           "reasoning_summary": "p4 introduces VPP"})

    monkeypatch.setattr(MCPToolClient, "run_agent", run_agent)
    baseline = StubBaseline()
    baseline.last_route = "verified_facts"
    verifier = VerifierModel({"supported_ids": ["p4-intro"], "unsupported_ids": ["p4-related"],
                              "notes": "the related-work passage is not about the introduction"})
    baseline.refinement_model = verifier
    result = AgenticResearch(baseline, agent_model=HighModel()).run(
        "Which paper introduced Video Prediction Policy?", "slow", "r-verify", concept=CONCEPT
    )
    run = result["run"]
    [request] = verifier.requests
    assert set(request["cited_evidence"]) == {"p4-intro", "p4-related"}
    assert "Video Prediction Policy" in request["cited_evidence"]["p4-intro"]
    assert "messages" not in request                       # fresh context: no agent conversation
    checks = {check.name: check for check in run.checks}
    assert checks["independent verification"].status == "warn"
    assert "p4-related" in checks["independent verification"].detail
    assert run.status == "answered" and run.execution_status != "degraded"   # advisory only
    verification = json.loads(result["artifacts"]["verification.json"])
    assert verification["independent_verification"]["supported_ids"] == ["p4-intro"]
    assert verification["independent_verification"]["unsupported_ids"] == ["p4-related"]
