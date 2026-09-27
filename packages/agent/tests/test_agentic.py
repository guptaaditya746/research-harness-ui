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
    # This corpus has no ranked retrieval, so grounding is the legacy phrase count: the reading
    # without support is shown last, not hidden (hiding needs ranked grounding).
    assert [r.label for r in run.readings] == ["Introducing paper", "Benchmark results", "Citation influence",
                                               "Quantum angle"]
    assert all(r.classes == ["Paper"] for r in run.readings)
    plans = json.loads(result["artifacts"]["plans.json"])
    assert plans[0]["evidence_hits"] == 4
    assert json.loads(result["artifacts"]["disagreement.json"])["hidden_ungrounded"] == []
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
        "used_for_retrieval": ["p4-intro"], "used_for_verification": ["p4-intro"], "findings": [],
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


def test_ungrounded_readings_stay_when_hiding_is_off(monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_HIDE_UNGROUNDED_READINGS", "false")
    baseline = StubBaseline()
    baseline.corpus = SearchCorpus()
    model = ConceptModel([{"concepts": [concept("Quantum angle", "quantum"), concept("Introducing paper"),
                                        concept("Benchmark results")]}])
    run = AgenticResearch(baseline, agent_model=model).run("Q?", "slow", "r-keep")["run"]
    assert [r.label for r in run.readings][-1] == "Quantum angle"


class RankedCorpus:
    """A corpus with ranked retrieval: the concept proposal retrieves before it proposes."""

    def __init__(self) -> None:
        self.passages = {
            "p1-a": {"passage_id": "p1-a", "paper_id": "p1", "text": "SafeRAG tests attack tasks such as silver noise."},
            "p1-b": {"passage_id": "p1-b", "paper_id": "p1", "text": "Security benchmarks for retrieval augmented generation."},
            "p2-a": {"passage_id": "p2-a", "paper_id": "p2", "text": "HydraRAG verifies evidence across sources."},
        }
        self.by_passage = self.passages
        self.queries: list[str] = []

    def hybrid_rank(self, query: str, scope=None, passage_ids=None, offset: int = 0, limit: int = 10) -> dict:
        self.queries.append(query)
        return {"items": list(self.passages.values())[:limit], "total": len(self.passages)}

    def keyphrases(self, scope=None, offset: int = 0, limit: int = 10) -> dict:
        return {"items": [{"phrase": "attack tasks"}, {"phrase": "silver noise"}]}

    def paper_title(self, paper_id: str) -> str:
        return {"p1": "SafeRAG", "p2": "HydraRAG"}[paper_id]


class RecordingConceptModel(ConceptModel):
    def complete_json(self, system: str, user: str) -> dict:
        self.user = json.loads(user)
        return super().complete_json(system, user)


def test_retrieve_first_shows_evidence_and_grounds_by_words_not_exact_phrases() -> None:
    baseline = StubBaseline()
    baseline.corpus = RankedCorpus()
    model = RecordingConceptModel([{"concepts": [
        {"label": "Attack tasks in SafeRAG", "gloss": "Which attacks are tested.", "classes": ["Paper"],
         "search_terms": ["attack task"], "evidence_ids": ["p1-a", "p9-invented"]},
        {"label": "Security benchmark scope", "gloss": "What the benchmark covers.", "classes": ["Paper"],
         "search_terms": ["security benchmark"]},
        {"label": "Quantum annealing", "gloss": "Unrelated.", "classes": ["Paper"],
         "search_terms": ["quantum annealing"]},
    ]}])
    result = AgenticResearch(baseline, agent_model=model).run("What does SafeRAG test?", "slow", "r-rf")
    assert [row["passage_id"] for row in model.user["passages"]] == ["p1-a", "p1-b", "p2-a"]
    assert model.user["keyphrases"] == ["attack tasks", "silver noise"]
    assert list(model.user)[-1] == "question"
    run = result["run"]
    assert [c.label for c in run.concepts] == ["Attack tasks in SafeRAG", "Security benchmark scope"]
    first = run.concepts[0]
    # "attack task" matches "attack tasks" (no exact phrase needed); the invented id is dropped.
    assert first.sample_sources[0] == "p1-a" and "p9-invented" not in first.sample_sources
    assert run.concepts[1].evidence_hits == 1
    assert json.loads(result["artifacts"]["disagreement.json"])["evidence_shown"] == ["p1-a", "p1-b", "p2-a"]


def test_research_starts_from_the_reading_s_supporting_passages() -> None:
    baseline = StubBaseline()
    baseline.corpus = RankedCorpus()
    runner = AgenticResearch(baseline, agent_model=HighModel())
    context = runner._grounded_context(None, {**CONCEPT, "sample_sources": ["p1-a", "missing"]})
    assert context["concept_evidence"] == [
        {"passage_id": "p1-a", "paper_id": "p1", "excerpt": "SafeRAG tests attack tasks such as silver noise."}]
    assert "p1-a" in runner._evidence_ids(context)


class CorpusWithPassages:
    def __init__(self, ids: list[str]) -> None:
        self.by_passage = {pid: {"passage_id": pid, "paper_id": pid.split("-")[0], "text": pid} for pid in ids}

    def paper_title(self, paper_id: str) -> str:
        return paper_id


def test_findings_reach_the_pipeline_one_passage_per_finding_first(monkeypatch) -> None:
    monkeypatch.setattr(AgenticResearch, "_heavy_queue_depth", staticmethod(lambda: 0))
    monkeypatch.setenv("HARNESS_AGENTIC_MAX_MCP_SEEDS", "3")
    ids = ["p1-a", "p1-b", "p1-c", "p2-a", "p3-a"]

    class FindingBaseline(StubBaseline):
        def __init__(self):
            self.seed_passage_ids = []
            self.agent_findings = []
            self.corpus = CorpusWithPassages(ids)

        def run(self, question, budget, run_id, concept=None):
            self.seen = (list(self.seed_passage_ids), list(self.agent_findings))
            return super().run(question, budget, run_id, concept=concept)

    def run_agent(self, model, question, grounded_context):
        self.sink({"call_id": "tool-001", "agent": "supervisor", "tool": "search_evidence",
                   "arguments": {"query": "q"}, "status": "ok", "duration_ms": 1.0, "result_count": 5,
                   "result": {"items": [{"passage_id": pid} for pid in ids]}})
        # Recorded during research with the scratchpad tool, then one more in the final decision.
        self.notes.append({"id": "F1", "statement": "SafeRAG tests four attacks.", "evidence_ids": ["p1-b", "p1-c"]})
        return json.dumps({"verified_claim_ids": [], "additional_evidence_ids": ["p1-a", "p2-a"],
                           "reasoning_summary": "found",
                           "findings": [{"statement": "HydraRAG verifies sources.", "evidence_ids": ["p3-a", "p9-x"]}]})

    monkeypatch.setattr(MCPToolClient, "run_agent", run_agent)
    baseline = FindingBaseline()
    result = AgenticResearch(baseline, agent_model=HighModel()).run("Compare them.", "slow", "r-f", concept=CONCEPT)
    seeds, findings = baseline.seen
    assert seeds == ["p1-b", "p3-a", "p2-a"]            # one per finding first, then another paper
    assert [item["evidence_ids"] for item in findings] == [["p1-b", "p1-c"], ["p3-a"]]
    assert baseline.agent_findings == [] and baseline.seed_passage_ids == []   # restored after the run
    ledger = json.loads(result["artifacts"]["research_ledger.json"])
    assert [item["statement"] for item in ledger["findings"]] == ["SafeRAG tests four attacks.", "HydraRAG verifies sources."]


def test_grounding_tolerates_plurals_of_short_words() -> None:
    from harness_agent.concepts import _content_tokens, _supports

    words = {"llms", "act", "as", "autonomous", "agents", "with", "tools"}
    assert _supports(_content_tokens("LLM agents"), words)
    assert _supports(_content_tokens("tool use"), words | {"uses"})
    assert not _supports(_content_tokens("tool planning"), words)


def test_legacy_grounding_keeps_every_reading(monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_CONCEPT_GROUNDING", "legacy")

    class BothCorpus(RankedCorpus, SearchCorpus):
        """Ranked retrieval is available, but legacy mode must not use it or hide readings."""

    baseline = StubBaseline()
    baseline.corpus = BothCorpus()
    model = ConceptModel([{"concepts": [concept("Quantum angle", "quantum"), concept("Introducing paper"),
                                        concept("Benchmark results")]}])
    run = AgenticResearch(baseline, agent_model=model).run("Q?", "slow", "r-legacy")["run"]
    assert "Quantum angle" in [r.label for r in run.readings]
    assert baseline.corpus.queries == []


def test_reader_findings_do_not_push_out_the_supervisor_evidence(monkeypatch) -> None:
    from time import monotonic

    from harness_agent.research import _Session

    monkeypatch.setenv("HARNESS_AGENTIC_MAX_MCP_SEEDS", "6")
    ids = [f"p{paper}-{index}" for paper in (1, 5, 7) for index in range(8)]
    baseline = StubBaseline()
    baseline.corpus = CorpusWithPassages(ids)
    runner = AgenticResearch(baseline, agent_model=HighModel())
    session = _Session(question="q", budget="slow", run_id="r", concept=None, concept_input={}, started=monotonic())
    session.observed = set(ids)
    session.agent_decision = {"additional_evidence_ids": ["p5-7", "p1-7"]}
    session.findings = [{"statement": f"reader finding {paper}-{index}", "evidence_ids": [f"p{paper}-{index}"],
                         "source": f"reader:p{paper}"} for paper in (5, 1, 7) for index in range(4)]
    runner._select_seeds(session)
    assert session.selected_ids[:2] == ["p5-7", "p1-7"]           # the supervisor's choice stays first
    assert "p7-0" in session.selected_ids                          # one passage from the third reader paper
    assert len({pid.split("-")[0] for pid in session.selected_ids}) == 3


def test_scalar_ids_in_the_decision_do_not_fail_the_run(monkeypatch) -> None:
    from time import monotonic

    from harness_agent.research import _Session

    baseline = StubBaseline()
    baseline.corpus = CorpusWithPassages(["p1-a", "p2-a"])
    runner = AgenticResearch(baseline, agent_model=HighModel())
    session = _Session(question="q", budget="slow", run_id="r", concept=None, concept_input={}, started=monotonic())
    session.observed = {"p1-a", "p2-a"}
    session.agent_decision = {"additional_evidence_ids": 7}
    runner._select_seeds(session)
    assert session.selected_ids == []


def test_refused_tool_calls_are_not_corpus_calls() -> None:
    from time import monotonic

    from harness_agent.research import _Session

    runner = AgenticResearch(StubBaseline(), agent_model=HighModel())
    session = _Session(question="q", budget="slow", run_id="r", concept=None, concept_input={}, started=monotonic())
    session.tool_calls = [{"tool": "rank_evidence", "status": "unavailable", "result": {"error": "not available"}},
                          {"tool": "search_evidence", "status": "ok", "result": {"items": [{"passage_id": "p1-a"}]}}]
    runner._collect_findings(session)
    assert [call["tool"] for call in session.corpus_calls] == ["search_evidence"]


def test_a_reading_answered_from_verified_facts_skips_the_research_agent() -> None:
    from harness_baseline.planner import QueryPlan

    class Planner:
        calls = 0

        def plan(self, question: str, concept: dict | None = None) -> QueryPlan:
            Planner.calls += 1
            return QueryPlan(operation="paper_authors", papers=["Some Paper"], rationale="authors of one paper")

    baseline = StubBaseline()
    baseline.planner = Planner()
    baseline.preset_plan = None
    runner = AgenticResearch(baseline, agent_model=HighModel(), high_model=HighModel())

    def no_agent(_session) -> None:
        raise AssertionError("the research agent must not start for a fact lookup")

    runner._mcp_research = no_agent
    runner._read_papers = no_agent
    result = runner.run("Who wrote Some Paper?", "slow", "r-route", concept=CONCEPT)
    checks = {check.name: check for check in result["run"].checks}
    assert checks["agent research (MCP)"].status == "skipped"
    assert "paper_authors" in checks["agent research (MCP)"].detail
    assert result["run"].status == "answered" and result["run"].execution_status == "completed"
    assert Planner.calls == 1 and isinstance(baseline.preset_plan, QueryPlan)   # handed to the pipeline


def test_a_research_reading_still_runs_the_agent() -> None:
    from harness_baseline.planner import QueryPlan

    class Planner:
        def plan(self, question: str, concept: dict | None = None) -> QueryPlan:
            return QueryPlan(operation="research", rationale="needs full text")

    baseline = StubBaseline()
    baseline.planner = Planner()
    runner = AgenticResearch(baseline, agent_model=HighModel(), high_model=HighModel())
    started: list[str] = []
    runner._mcp_research = lambda session: started.append("mcp")
    runner.run("How does Some Paper work?", "slow", "r-route-research", concept=CONCEPT)
    assert started == ["mcp"]


def test_unstructured_decision_with_recorded_findings_is_not_degraded(monkeypatch) -> None:
    monkeypatch.setattr(AgenticResearch, "_heavy_queue_depth", staticmethod(lambda: 0))

    def run_agent(self, model, question, grounded_context):
        self.sink({
            "call_id": "tool-001", "agent": "supervisor", "tool": "search_evidence",
            "arguments": {"query": "NLP task"}, "status": "ok", "duration_ms": 3.0, "result_count": 1,
            "result": {"items": [{"passage_id": "p4-intro", "paper_id": "p4", "text": "Relevant task."}]},
        })
        self.notes.append({"id": "F1", "statement": "p4 studies an NLP task.", "evidence_ids": ["p4-intro"]})
        return "The corpus includes several NLP tasks."

    monkeypatch.setattr(MCPToolClient, "run_agent", run_agent)
    result = AgenticResearch(StubBaseline(), agent_model=HighModel()).run(
        "Which NLP tasks appear most often?", "slow", "r-unstructured-findings", concept=CONCEPT,
    )
    assert result["run"].execution_status == "completed"
    decision = next(check for check in result["run"].checks if check.name == "agent decision")
    assert decision.status == "warn" and "1 recorded findings" in decision.detail


def test_routing_needs_resolved_papers_and_reports_a_fallback_honestly() -> None:
    from harness_baseline.planner import QueryPlan

    class Planner:
        def plan(self, question: str, concept: dict | None = None) -> QueryPlan:
            return QueryPlan(operation="paper_authors", papers=["Unknown Paper"])

    class Unresolved(StubBaseline):
        preset_plan = None

        def _resolve_papers(self, plan):
            return [], list(plan.papers)

    baseline = Unresolved()
    baseline.planner = Planner()
    runner = AgenticResearch(baseline, agent_model=HighModel(), high_model=HighModel())
    started: list[str] = []
    runner._mcp_research = lambda session: started.append("mcp")
    runner.run("Who wrote Unknown Paper?", "slow", "r-route-unresolved", concept=CONCEPT)
    assert started == ["mcp"]                        # an unresolved paper is not a fact lookup

    class FellBack(StubBaseline):
        preset_plan = None
        last_route = "research"

        def _resolve_papers(self, plan):
            return ["p4"], []

    baseline = FellBack()
    baseline.planner = Planner()
    runner = AgenticResearch(baseline, agent_model=HighModel(), high_model=HighModel())
    runner._mcp_research = lambda session: (_ for _ in ()).throw(AssertionError("agent must not start"))
    run = runner.run("Who wrote Some Paper?", "slow", "r-route-fellback", concept=CONCEPT)["run"]
    assert run.execution_status == "degraded"
    research = next(check for check in run.checks if check.name == "agent research (MCP)")
    assert "did not cover it" in research.detail


def test_every_reading_of_a_fact_question_gets_the_plan_of_the_question() -> None:
    from harness_baseline.planner import QueryPlan, QueryPlanner

    calls: list[str] = []

    class Model:
        model = "stub-planner"

        def complete_json(self, system: str, user: str) -> dict:
            calls.append(user)
            if "research_interpretation" in user:            # a reading-specific plan would differ
                return {"operation": "research", "rationale": "reading says research"}
            return {"operation": "paper_authors", "papers": ["Some Paper"], "rationale": "authors"}

    QueryPlanner._question_plans.clear()
    baseline = StubBaseline()
    baseline.planner = QueryPlanner(Model(), profile={"papers": 1, "venues": {"ICML 2024": 1}, "years": {"2024": 1}})
    baseline.preset_plan = None
    runner = AgenticResearch(baseline, agent_model=HighModel(), high_model=HighModel())
    runner._mcp_research = lambda session: (_ for _ in ()).throw(AssertionError("agent must not start"))
    for index in range(3):
        runner.run("Who wrote Some Paper?", "slow", f"r-reading-{index}", concept=CONCEPT)
        assert isinstance(baseline.preset_plan, QueryPlan) and baseline.preset_plan.operation == "paper_authors"
    assert len(calls) == 1                                        # planned once, reused for every reading
    assert '"corpus"' in calls[0] and "ICML 2024" in calls[0]
