import json
from pathlib import Path

from harness_baseline import Baseline20
from harness_baseline.planner import PaperResolver, QueryPlan, QueryPlanner
from harness_corpus.store import CorpusStore
from harness_extraction import ExtractedTriple, ExtractionEngine, SchemaInference
from harness_schema.models import required_artifacts
from harness_semantic import hand_written_subschema


class RecordedModel:
    model = "recorded"

    def __init__(self) -> None:
        self.replies = iter(
            [
                {
                    "label": "Introduced method",
                    "gloss": "Paper proposing VPP",
                    "classes": ["Paper", "Claim"],
                    "properties": ["claims"],
                },
                {
                    "triples": [
                        {
                            "subject": "p4",
                            "predicate": "claims",
                            "object": "Video Prediction Policy",
                            "source_id": "p4-abstract",
                            "evidence": "we propose the Video Prediction Policy",
                        }
                    ]
                },
                {
                    "sparql": "PREFIX hrn: <https://harness.local/schema#>\nSELECT ?paper ?claim WHERE { ?paper hrn:claims ?claim . }",
                    "answer": "Paper p4 introduces Video Prediction Policy [p4-abstract].",
                    "cited_source_ids": ["p4-abstract"],
                },
            ]
        )

    def complete_json(self, system: str, user: str) -> dict:
        json.loads(user)
        return next(self.replies)


class StubPlanner:
    """Stands in for the model planner: maps a question to a fixed typed plan."""

    def __init__(self, plans: dict[str, QueryPlan]) -> None:
        self.plans = plans

    def plan(self, question: str, concept: dict | None = None) -> QueryPlan:
        return self.plans.get(question, QueryPlan(operation="research"))


def plan(operation: str, *papers: str, **fields) -> QueryPlan:
    return QueryPlan(operation=operation, papers=list(papers), **fields)


def corpus(tmp_path: Path) -> CorpusStore:
    rows = {
        "papers.jsonl": [{"paper_id": "p4", "title": "Video Prediction Policy", "year": 2025}],
        "passages.jsonl": [
            {
                "passage_id": "p4-abstract",
                "paper_id": "p4",
                "section_path": "Abstract",
                "para_idx": 0,
                "text": "We propose the Video Prediction Policy for robot learning.",
            }
        ],
        "facts.jsonl": [
            {
                "fact_id": "p4-metadata-0",
                "subject": "p4",
                "predicate": "title",
                "object": "Video Prediction Policy",
                "source_id": "p4-metadata",
            }
        ],
        "references.jsonl": [],
    }
    for name, values in rows.items():
        (tmp_path / name).write_text("".join(json.dumps(row) + "\n" for row in values))
    return CorpusStore(tmp_path)


def test_baseline_runs_four_nodes_and_writes_complete_artifact_set(tmp_path: Path) -> None:
    question = "Which paper introduced Video Prediction Policy?"
    runner = Baseline20(
        corpus(tmp_path), ExtractionEngine(RecordedModel(), hand_written_subschema()),
        planner=StubPlanner({question: plan("find_paper", "Video Prediction Policy")}),
    )
    result = runner.run(question, run_id="r-live")

    assert result["run"].status == "answered"
    assert [row.action for row in result["run"].trace] == [
        "Understanding question",
        "Retrieving verified corpus facts",
        "Selecting atomic triples",
    ]
    assert set(result["artifacts"]) | {"run.json"} == set(required_artifacts("baseline"))
    assert "p4-metadata" in result["run"].readings[0].answer
    assert [(item.s, item.p, item.o) for item in result["run"].readings[0].triples] == [
        ("p4", "title", "Video Prediction Policy")
    ]
    calls = [json.loads(line) for line in result["artifacts"]["tool_calls.jsonl"].splitlines()]
    assert [(call["tool"], call["result_count"]) for call in calls] == [("facts", 1)]


def test_proposed_schema_relation_reaches_graph_and_registry(tmp_path: Path) -> None:
    question = "Which task does Video Prediction Policy address?"

    class SchemaModel:
        def complete_json(self, system: str, user: str) -> dict:
            return {
                "label": "Addressed task", "gloss": "Paper and task",
                "classes": ["Paper", "Task"], "properties": ["studiesTask"],
                "class_definitions": [{"name": "Task", "definition": "A research task."}],
                "property_definitions": [{"name": "studiesTask", "definition": "Paper studies a task.",
                                          "domain": "Paper", "range": "Task"}],
            }

    class AnswerModel:
        def __init__(self):
            self.replies = iter([
                {"triples": [{"subject": "p4", "predicate": "studiesTask",
                              "object": "robot learning", "source_id": "p4-abstract",
                              "evidence": "for robot learning"}]},
                {"sparql": "PREFIX hrn: <https://harness.local/schema#>\n"
                           "SELECT ?paper ?task WHERE { ?paper hrn:studiesTask ?task . }",
                 "answer": "Video Prediction Policy addresses robot learning [p4-abstract].",
                 "cited_source_ids": ["p4-abstract"]},
            ])

        def complete_json(self, system: str, user: str) -> dict:
            return next(self.replies)

    proposed = []
    runner = Baseline20(
        corpus(tmp_path),
        ExtractionEngine(AnswerModel(), hand_written_subschema(), schema_model=SchemaModel()),
        planner=StubPlanner({question: plan("research")}),
        relation_proposer=lambda *args: proposed.append(args),
    )

    result = runner.run(question, run_id="r-proposed-task")

    assert result["run"].status == "answered"
    assert result["run"].readings[0].triples[0].p == "studiesTask"
    assert proposed == [("studiesTask", "Paper studies a task.", "r-proposed-task", "Paper", "Task")]
    assert b"studiesTask" in result["artifacts"]["subschema.ttl"]


def test_baseline_no_hits_is_a_completed_semantic_no_evidence_run(tmp_path: Path) -> None:
    runner = Baseline20(
        corpus(tmp_path), ExtractionEngine(RecordedModel(), hand_written_subschema())
    )
    result = runner.run("Unfindable zxqvterm", run_id="r-failed")

    assert result["run"].status == "no_evidence_found"
    assert "No evidence was found" in (result["run"].reason or "")
    assert result["run"].search_coverage["coverage_complete"] is False
    assert set(result["artifacts"]) | {"run.json"} == set(required_artifacts("baseline"))


def test_agent_seed_passage_survives_question_ranking(tmp_path: Path) -> None:
    runner = Baseline20(
        corpus(tmp_path), ExtractionEngine(RecordedModel(), hand_written_subschema()),
        enhanced=True, seed_passage_ids=["p4-abstract"],
    )
    state = runner._retrieve({
        "question": "Unfindable zxqvterm", "plan": QueryPlan(operation="research"),
        "paper_ids": [], "concept": None,
    })

    assert [passage["passage_id"] for passage in state["passages"]] == ["p4-abstract"]


def test_agentic_passage_cap_keeps_agent_evidence_and_diverse_papers(tmp_path: Path) -> None:
    store = corpus(tmp_path)
    for index in range(20):
        paper_id = f"p{index + 10}"
        paper = {"paper_id": paper_id, "title": f"Task paper {index}", "year": 2025}
        passage = {
            "passage_id": f"{paper_id}-abstract", "paper_id": paper_id,
            "section_path": "Abstract", "para_idx": 0,
            "text": f"NLP tasks and approaches occur in this paper; classification approach {index}.",
        }
        store.papers.append(paper)
        store.by_paper[paper_id] = paper
        store.passages.append(passage)
        store.by_passage[passage["passage_id"]] = passage
    runner = Baseline20(
        store, ExtractionEngine(RecordedModel(), hand_written_subschema()),
        enhanced=True, max_passages=6, seed_passage_ids=["p4-abstract"],
    )
    runner.started = __import__("time").monotonic()

    state = runner._retrieve({
        "question": "Which NLP tasks and approaches occur?",
        "plan": QueryPlan(operation="research"), "paper_ids": [], "concept": None,
    })

    passages = state["passages"]
    assert len(passages) == 6
    assert passages[0]["passage_id"] == "p4-abstract"
    assert len({row["paper_id"] for row in passages}) == 6
    assert state["retrieval"]["selected"] == 6


def test_enhanced_baseline_seeds_structural_facts_and_batches_extraction(tmp_path: Path) -> None:
    question = "Which paper introduced Video Prediction Policy?"
    runner = Baseline20(
        corpus(tmp_path), ExtractionEngine(RecordedModel(), hand_written_subschema()), enhanced=True,
        planner=StubPlanner({question: plan("find_paper", "Video Prediction Policy")}),
    )
    result = runner.run(question, run_id="r-enhanced")

    assert result["run"].path == "baseline-enhanced"
    assert "retrieval.json" in result["artifacts"]
    assert "extraction_batches.json" in result["artifacts"]
    assert "corpus-seed" in result["artifacts"]["graph.trig"].decode()
    assert set(result["artifacts"]) | {"run.json"} == set(required_artifacts("baseline-enhanced"))


def test_missing_incoming_citations_do_not_return_unrelated_triples(tmp_path: Path) -> None:
    question = "Which papers cite Video Prediction Policy?"
    runner = Baseline20(
        corpus(tmp_path), ExtractionEngine(RecordedModel(), hand_written_subschema()),
        planner=StubPlanner({question: plan("citing_papers", "Video Prediction Policy")}),
    )
    result = runner.run(question, run_id="r-citations")

    assert result["run"].status == "corpus_absent"
    assert result["run"].evidence_surface == "citation_index"
    assert result["run"].search_coverage["exhaustive"] is True
    assert not result["run"].readings
    assert "No in-corpus evidence found" in result["artifacts"]["answer.md"].decode()


def test_citation_absence_survives_question_paraphrase(tmp_path: Path) -> None:
    # Different wording, same meaning: the plan, not the phrasing, selects the operation.
    question = "Is Video Prediction Policy referenced by anything else in this collection?"
    runner = Baseline20(
        corpus(tmp_path), ExtractionEngine(RecordedModel(), hand_written_subschema()),
        planner=StubPlanner({question: plan("citing_papers", "the Video Prediction Policy paper")}),
    )
    run = runner.run(question)["run"]
    assert run.status == "corpus_absent"
    assert run.search_coverage["matches"] == 0


def test_preprint_and_journal_version_count_as_one_work(tmp_path: Path) -> None:
    store = corpus(tmp_path)
    store.by_paper["p5"] = {"paper_id": "p5", "title": "Video Prediction Policy", "year": 2025}
    store.by_paper["p6"] = {"paper_id": "p6", "title": "A Different Robot Policy", "year": 2026}
    runner = Baseline20(store, ExtractionEngine(RecordedModel(), hand_written_subschema()))
    rows = [{"subject": paper_id} for paper_id in ("p4", "p5", "p6")]

    assert [row["subject"] for row in runner._distinct_works(rows)] == ["p4", "p6"]


def test_the_graph_query_is_built_from_the_cited_triples_not_the_model(tmp_path: Path) -> None:
    class MalformedQueryModel:
        def complete_json(self, system: str, user: str) -> dict:
            return {
                "sparql": "SELECT ?x WHERE { p4 reportsMetric ?x }",
                "answer": "VPP reports 41.5% improvement [p4-abstract].",
                "cited_source_ids": ["p4-abstract"],
            }

    runner = Baseline20(corpus(tmp_path), ExtractionEngine(MalformedQueryModel(), hand_written_subschema()))
    runner.started = __import__("time").monotonic()
    state = {
        "question": "Describe the VPP result.",
        "inferred": SchemaInference(label="Metric", gloss="Result", classes=["Metric"], properties=["reportsMetric"]),
        "extracted": [ExtractedTriple(subject="p4", predicate="reportsMetric", object="41.5%", source_id="p4-abstract", evidence="41.5% improvement")],
        "passages": [],
    }

    answer = runner._query_answer(state)["answer"]

    # The model's malformed query is ignored: the query selects the cited triples' predicates.
    assert "?subject hrn:reportsMetric ?object" in answer.sparql and "p4 reportsMetric" not in answer.sparql
    assert "41.5%" in answer.answer
    assert not any(row.action == "Repairing graph query" for row in runner.trace)


def test_year_and_venue_use_verified_corpus_facts(tmp_path: Path) -> None:
    store = corpus(tmp_path)
    store.fact_rows.extend([
        {"fact_id": "p4-year", "subject": "p4", "predicate": "year", "object": "2025", "source_id": "p4-metadata"},
        {"fact_id": "p4-venue", "subject": "p4", "predicate": "venue", "object": "Advanced Robotics", "source_id": "p4-metadata"},
    ])
    runner = Baseline20(store, ExtractionEngine(RecordedModel(), hand_written_subschema()), planner=StubPlanner({
        "When did Video Prediction Policy come out?": plan("paper_year", "Video Prediction Policy"),
        "Where did Video Prediction Policy appear?": plan("paper_venue", "Video Prediction Policy"),
    }))

    year = runner.run("When did Video Prediction Policy come out?")["run"]
    venue = runner.run("Where did Video Prediction Policy appear?")["run"]

    assert year.status == venue.status == "answered"
    assert [(item.p, item.o) for item in year.readings[0].triples] == [("year", "2025")]
    assert [(item.p, item.o) for item in venue.readings[0].triples] == [("venue", "Advanced Robotics")]


def test_resolved_citation_count_and_year_aggregation(tmp_path: Path) -> None:
    store = corpus(tmp_path)
    store.by_paper["p5"] = {"paper_id": "p5", "title": "Another Robot Paper", "year": 2025}
    store.fact_rows.extend([
        {"fact_id": "p4-year", "subject": "p4", "predicate": "year", "object": "2025", "source_id": "p4-metadata"},
        {"fact_id": "p5-year", "subject": "p5", "predicate": "year", "object": "2025", "source_id": "p5-metadata"},
    ])
    store.references.append({"reference_id": "p5-ref-1", "paper_id": "p5", "title": "Video Prediction Policy", "resolved_paper_id": "p4"})
    runner = Baseline20(store, ExtractionEngine(RecordedModel(), hand_written_subschema()), planner=StubPlanner({
        "How many indexed papers cite Video Prediction Policy?": plan("citing_count", "Video Prediction Policy"),
        "How many papers in the corpus were published in 2025?": plan("papers_in_year", year=2025),
    }))

    count = runner.run("How many indexed papers cite Video Prediction Policy?")["run"]
    years = runner.run("How many papers in the corpus were published in 2025?")["run"]

    assert count.status == years.status == "answered"
    assert "1 indexed papers cite" in count.readings[0].answer
    assert [(item.p, item.o) for item in count.readings[0].triples] == [("cites", "p4")]
    assert "2 corpus papers" in years.readings[0].answer
    assert len(years.readings[0].triples) == 2


def test_named_paper_retrieval_ranks_target_evidence_without_unrelated_papers(tmp_path: Path) -> None:
    store = corpus(tmp_path)
    store.papers.append({"paper_id": "p5", "title": "Unrelated Robot Policy", "year": 2025})
    store.by_paper["p5"] = store.papers[-1]
    store.passages.extend([
        {"passage_id": "p4-result", "paper_id": "p4", "text": "The CALVIN ABC-D benchmark improved by 41.5%."},
        {"passage_id": "p5-result", "paper_id": "p5", "text": "The CALVIN ABC-D benchmark improved by 99%."},
    ])
    store.by_passage.update({row["passage_id"]: row for row in store.passages})
    runner = Baseline20(store, ExtractionEngine(RecordedModel(), hand_written_subschema()), enhanced=True)
    runner.started = __import__("time").monotonic()

    state = runner._retrieve({"question": "What CALVIN ABC-D improvement did Video Prediction Policy report?"})

    assert state["retrieval"]["mentioned_paper_ids"] == ["p4"]
    assert "p4-result" in {row["passage_id"] for row in state["passages"]}
    assert all(row["paper_id"] == "p4" for row in state["passages"])


def test_baseline_never_pauses_for_canned_interpretations(tmp_path: Path) -> None:
    """Interpretation is the agent's job; baseline answers the question as asked."""
    store = corpus(tmp_path)
    store.fact_rows.append({
        "fact_id": "p4-author", "subject": "p4", "predicate": "author",
        "object": "Ada Scholar", "source_id": "p4-metadata",
    })
    runner = Baseline20(store, ExtractionEngine(RecordedModel(), hand_written_subschema()))

    run = runner.run("Who is the best author in the corpus?")["run"]

    assert run.status != "awaiting_reading"
    assert all(row.action != "Clarification required" for row in run.trace)


def test_concept_widens_retrieval_and_reaches_the_model(tmp_path: Path) -> None:
    class CapturingModel(RecordedModel):
        def __init__(self) -> None:
            super().__init__()
            self.inputs: list[str] = []

        def complete_json(self, system: str, user: str) -> dict:
            self.inputs.append(user)
            return super().complete_json(system, user)

    model = CapturingModel()
    runner = Baseline20(corpus(tmp_path), ExtractionEngine(model, hand_written_subschema()),
                        enhanced=True)
    concept = {"label": "Introduced method", "gloss": "The method a paper proposes.",
               "classes": ["Paper"], "search_terms": ["video prediction"]}
    question = "Which methods support robot learning?"

    result = runner.run(question, concept=concept)

    assert result["run"].question == question
    retrieval = json.loads(result["artifacts"]["retrieval.json"])
    assert '"video prediction"' in [item["query"] for item in retrieval["queries"]]
    assert model.inputs and all(
        "Research interpretation chosen by the user: Introduced method" in item
        for item in model.inputs
    )


# ─── Question understanding ──────────────────────────────────────────────────

class JsonModel:
    def __init__(self, payload) -> None:
        self.payload = payload
        self.inputs: list[str] = []

    def complete_json(self, system: str, user: str) -> dict:
        self.inputs.append(user)
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


def test_planner_accepts_a_valid_typed_plan_and_passes_the_concept() -> None:
    model = JsonModel({"operation": "cites", "papers": ["RISE", "Dream to Manipulate"],
                       "evidence_focus": "nonsense", "rationale": "citation check"})
    result = QueryPlanner(model).plan("Is Dream to Manipulate in RISE's references?",
                                      {"label": "Citation", "gloss": "Direct reference."})
    assert result.operation == "cites"
    assert result.papers == ["RISE", "Dream to Manipulate"]
    assert result.evidence_focus == "general"
    assert result.source == "model"
    assert "Direct reference." in model.inputs[0]


def test_planner_degrades_to_research_instead_of_guessing() -> None:
    assert QueryPlanner(JsonModel({"operation": "delete_everything"})).plan("q").operation == "research"
    missing = QueryPlanner(JsonModel({"operation": "cites", "papers": ["Only one"]})).plan("q")
    assert missing.operation == "research" and missing.source == "fallback"
    no_year = QueryPlanner(JsonModel({"operation": "papers_in_year"})).plan("q")
    assert no_year.operation == "research"
    down = QueryPlanner(JsonModel(RuntimeError("proxy down"))).plan("q")
    assert down.operation == "research" and "unavailable" in down.rationale


def test_a_run_whose_triples_cannot_enter_the_graph_is_no_results_not_an_empty_answer(tmp_path: Path) -> None:
    class UnindexedTarget:
        model = "recorded"

        def __init__(self) -> None:
            self.replies = iter([
                {"label": "Compared methods", "gloss": "What VPP is compared to", "classes": ["Paper"],
                 "properties": ["comparesTo"]},
                {"triples": [{"subject": "p4", "predicate": "comparesTo", "object": "A Paper Outside The Corpus",
                              "source_id": "p4-abstract", "evidence": "We propose the Video Prediction Policy"}]},
                {"answer": "p4 is compared to another paper [p4-abstract].", "cited_source_ids": ["p4-abstract"]},
            ])

        def complete_json(self, system: str, user: str) -> dict:
            return next(self.replies)

    question = "What is the Video Prediction Policy compared to?"
    runner = Baseline20(corpus(tmp_path), ExtractionEngine(UnindexedTarget(), hand_written_subschema()),
                        planner=StubPlanner({question: QueryPlan(operation="research")}))
    run = runner.run(question)["run"]
    assert run.status != "answered" and run.reason == "no_results"
    failed = [check.detail for check in run.checks if check.status == "fail"]
    assert any("1 paper not indexed" in detail for detail in failed)


def test_a_fact_plan_must_take_its_arguments_from_the_question() -> None:
    # A term copied from the reading's label, not the question, answers a different question.
    concept = {"label": "MethodX robustness framework", "gloss": "Frameworks that test robustness."}
    borrowed = QueryPlanner(JsonModel({"operation": "papers_with_title_term", "term": "robustness framework"})).plan(
        "Who wrote the MethodX paper?", concept)
    assert borrowed.operation == "research" and borrowed.source == "fallback"
    assert "term 'robustness framework'" in borrowed.rationale
    assert borrowed.stated_measure() == "MethodX robustness framework"
    wrong_year = QueryPlanner(JsonModel({"operation": "papers_in_year", "year": 2023})).plan("List the papers from 2024.")
    assert wrong_year.operation == "research" and "year 2023" in wrong_year.rationale
    wrong_venue = QueryPlanner(JsonModel({"operation": "papers_in_venue", "venues": ["COLT 2021"]})).plan(
        "List every COLT paper.")
    assert wrong_venue.operation == "research" and "venue 'COLT 2021'" in wrong_venue.rationale
    # Filler and stop words in a venue, "ICML2024" as one word, a year inside the venue.
    for venue, year in (("Findings of ICML 2024", None), ("ICML 2024 main conference", 2024)):
        plan = QueryPlanner(JsonModel({"operation": "papers_in_venue", "venues": [venue], "year": year})).plan(
            "Show every ICML2024 Findings paper.")
        assert plan.operation == "papers_in_venue", plan.rationale
    plural = QueryPlanner(JsonModel({"operation": "papers_with_title_term", "term": "transformer", "year": 2022})).plan(
        "Show the 2022 titles that contain the word transformers.")
    assert plural.operation == "papers_with_title_term"
    loose = QueryPlanner(JsonModel({"operation": "papers_with_title_term", "term": "findings"})).plan(
        "Help me find titles about graphs.")
    assert loose.operation == "research"                  # "find" in the question does not state "findings"
    named = QueryPlanner(JsonModel({"operation": "paper_year", "papers": ["BenchY"], "venues": ["NeurIPS 2023"]})).plan(
        "When did the NeurIPS 2023 BenchY paper come out?", concept)
    assert named.operation == "paper_year"                # the venue only describes the named paper


def test_a_fact_plan_that_leaves_out_a_condition_becomes_research() -> None:
    result = QueryPlanner(JsonModel({"operation": "papers_in_venue", "venues": ["ICML"],
                                     "ignored": ["report results on BenchY"]})).plan(
        "Show the ICML papers that report results on BenchY.")
    assert result.operation == "research" and "BenchY" in result.rationale
    research = QueryPlanner(JsonModel({"operation": "research", "ignored": ["x"]})).plan("Show the ICML papers on y.")
    assert research.operation == "research" and research.source == "model"
    none = QueryPlanner(JsonModel({"operation": "paper_authors", "papers": ["MethodX"], "ignored": ["none"]})).plan(
        "Who wrote MethodX?")
    assert none.operation == "paper_authors"
    # A filter the operation does not apply (the citing papers' year) is a condition left out.
    dated = QueryPlanner(JsonModel({"operation": "citing_count", "papers": ["MethodX"], "year": 2024})).plan(
        "How many 2024 papers cite MethodX?")
    assert dated.operation == "research" and "year" in dated.rationale


def test_paper_resolver_links_mentions_and_refuses_ambiguity() -> None:
    resolver = PaperResolver({
        "p6": "FOCUS: object-centric world models for robotic manipulation",
        "p47": "Dream to Manipulate: Compositional World Models Empowering Robot Imitation Learning with Imagination",
        "p28": "DayDreamer: World Models for Physical Robot Learning",
        "p4": "Video Prediction Policy: A Generalist Robot Policy with Predictive Visual Representations",
    })
    assert resolver.resolve("FOCUS") == "p6"
    assert resolver.resolve("Dream to Manipulate") == "p47"
    assert resolver.resolve("the DayDreamer paper") == "p28"
    assert resolver.resolve("video prediction policy") == "p4"
    assert resolver.resolve("world models") is None
    assert resolver.resolve("Attention Is All You Need") is None


def test_unresolved_paper_falls_back_to_full_text_research(tmp_path: Path) -> None:
    question = "Who wrote the Nonexistent Robot Paper?"
    runner = Baseline20(
        corpus(tmp_path), ExtractionEngine(RecordedModel(), hand_written_subschema()),
        planner=StubPlanner({question: plan("paper_authors", "Nonexistent Robot Paper")}),
    )
    run = runner.run(question)["run"]
    actions = [row.action for row in run.trace]
    assert "Paper not found in corpus titles" in actions
    assert "Retrieving verified corpus facts" not in actions


def test_author_ranking_and_threshold_come_from_the_plan(tmp_path: Path) -> None:
    store = corpus(tmp_path)
    store.by_paper["p5"] = {"paper_id": "p5", "title": "A Second Robot Paper", "year": 2025}
    store.fact_rows.extend([
        {"fact_id": "a1", "subject": "p4", "predicate": "author", "object": "Ada", "source_id": "p4-metadata"},
        {"fact_id": "a2", "subject": "p5", "predicate": "author", "object": "Ada", "source_id": "p5-metadata"},
        {"fact_id": "a3", "subject": "p5", "predicate": "author", "object": "Bo", "source_id": "p5-metadata"},
    ])
    runner = Baseline20(store, ExtractionEngine(RecordedModel(), hand_written_subschema()), planner=StubPlanner({
        "Who has the most papers here?": plan("author_paper_counts", rank="top"),
        "Which authors show up more than once?": plan("author_paper_counts", min_papers=2),
    }))
    top = runner.run("Who has the most papers here?")["run"]
    repeated = runner.run("Which authors show up more than once?")["run"]
    assert top.status == repeated.status == "answered"
    assert top.readings[0].answer.startswith("Ada appears on the most distinct corpus papers (2)")
    assert {item.o for item in repeated.readings[0].triples} == {"Ada"}


def _ranking_runner(tmp_path: Path, plans: dict[str, QueryPlan]) -> Baseline20:
    store = corpus(tmp_path)
    store.by_paper["p5"] = {"paper_id": "p5", "title": "A Second Robot Paper", "year": 2025}
    store.by_paper["p8"] = {"paper_id": "p8", "title": "A Citing Robot Paper", "year": 2026}
    store.fact_rows.extend([
        {"fact_id": "a1", "subject": "p4", "predicate": "author", "object": "Ada", "source_id": "p4-metadata"},
        {"fact_id": "a2", "subject": "p5", "predicate": "author", "object": "Ada", "source_id": "p5-metadata"},
        {"fact_id": "a3", "subject": "p5", "predicate": "author", "object": "Bo", "source_id": "p5-metadata"},
        {"fact_id": "a4", "subject": "p8", "predicate": "author", "object": "Cy", "source_id": "p8-metadata"},
    ])
    store.references.extend([
        {"reference_id": "p8-ref-1", "paper_id": "p8", "resolved_paper_id": "p4", "title": "Video Prediction Policy"},
        {"reference_id": "p8-ref-2", "paper_id": "p8", "resolved_paper_id": "p5", "title": "A Second Robot Paper"},
    ])
    store.by_reference = {row["reference_id"]: row for row in store.references}
    return Baseline20(store, ExtractionEngine(RecordedModel(), hand_written_subschema()),
                      planner=StubPlanner(plans))


def test_fact_answer_graph_shares_nodes_and_separates_computed_results(tmp_path: Path) -> None:
    question = "Who has the most papers here?"
    run = _ranking_runner(tmp_path, {question: plan("author_paper_counts", rank="top")}).run(question)["run"]
    reading = run.readings[0]
    graph = reading.graph
    assert graph is not None
    authors = [node for node in graph.nodes if node.kind == "author"]
    assert [node.label for node in authors] == ["Ada"]           # one node, shared by both papers
    assert sum(1 for edge in graph.edges if edge.target == authors[0].id and edge.kind == "fact") == 2
    [computed] = graph.computed
    assert computed.label == "2 distinct papers"
    fact_ids = {edge.id for edge in graph.edges if edge.kind == "fact"}
    assert set(computed.derived_from) <= fact_ids and len(computed.derived_from) == 2
    assert any(node.kind == "computed" for node in graph.nodes)
    assert reading.measure and "distinct corpus papers per author" in reading.measure
    names = {check.name: check.status for check in run.checks}
    assert names["question understanding"] == "pass"
    assert names["verified facts"] == "pass"
    assert names["retrieval"] == names["extraction"] == "skipped"
    assert names["triple provenance"] == "pass"


def test_author_citation_ranking_counts_citing_papers_with_cites_edges(tmp_path: Path) -> None:
    question = "Which author is cited most by other corpus papers?"
    run = _ranking_runner(tmp_path, {question: plan("author_citation_counts", rank="top")}).run(question)["run"]
    assert run.status == "answered"
    reading = run.readings[0]
    # p8 cites two of Ada's papers but is one citing paper; Bo ties with Ada, and ties are kept.
    assert reading.answer.startswith("Most cited in the corpus: Ada (1 citing paper); Bo (1 citing paper)")
    predicates = {triple.p for triple in reading.triples}
    assert {"cites", "author"} <= predicates or {"cites", "authoredBy"} <= predicates
    assert [item.label for item in reading.graph.computed] == ["1 citing paper", "1 citing paper"]
    ada = next(item for item in reading.graph.computed if item.about.endswith("ada"))
    derived = [edge for edge in reading.graph.edges if edge.id in ada.derived_from]
    assert sorted(edge.label for edge in derived).count("cites") == 2


def test_research_failure_marks_the_stage_that_ended_the_run(tmp_path: Path) -> None:
    runner = Baseline20(corpus(tmp_path), ExtractionEngine(RecordedModel(), hand_written_subschema()))
    run = runner.run("Unfindable zxqvterm")["run"]
    checks = {check.name: check for check in run.checks}
    assert checks["retrieval"].status == "fail"
    assert "extraction" not in checks or checks["extraction"].status != "pass"


def test_planner_states_the_measure_used() -> None:
    assert "distinct corpus papers citing" in plan("author_citation_counts", rank="top").stated_measure()
    research = QueryPlan(operation="research", measure="  success rate on CALVIN ABC→D ")
    assert research.stated_measure() == "success rate on CALVIN ABC→D"
    assert QueryPlan(operation="research").stated_measure() is None


def test_research_measure_follows_the_chosen_concept() -> None:
    model = JsonModel({"operation": "research", "measure": "recurring research themes",
                       "rationale": "needs full text"})
    concept = {"label": "Commonly used datasets", "gloss": "Datasets referenced by most papers."}
    result = QueryPlanner(model).plan("What research themes recur across this corpus?", concept)
    assert result.stated_measure() == "Commonly used datasets"
    fact = QueryPlanner(JsonModel({"operation": "author_paper_counts", "rank": "top"})).plan("q", concept)
    assert "distinct corpus papers per author" in fact.stated_measure()


def test_hybrid_mode_adds_dense_candidates_lexical_retrieval_misses(tmp_path: Path, monkeypatch) -> None:
    import numpy as np
    from harness_corpus import store as store_module
    from harness_corpus.dense import passages_digest

    store = corpus(tmp_path)
    folder = tmp_path / "index" / "dense"
    folder.mkdir(parents=True)
    np.save(folder / "embeddings.npy", np.asarray([[1.0, 0.0]], dtype=np.float16))
    (folder / "passage_ids.json").write_text(json.dumps(["p4-abstract"]))
    (folder / "meta.json").write_text(json.dumps({"model": "test", "passages_sha256": passages_digest(tmp_path)}))
    store = CorpusStore(tmp_path)
    store_module._embed_query_cached.cache_clear()
    store_module._embedding_down_until.clear()
    monkeypatch.setattr(store_module, "embed_texts", lambda texts, **_: np.asarray([[1.0, 0.0]], dtype=np.float32))
    runner = Baseline20(store, ExtractionEngine(RecordedModel(), hand_written_subschema()))
    # No word of the question appears in the passage; only the embedding links them.
    state = {"question": "Which approaches forecast footage for manipulators?",
             "plan": QueryPlan(operation="research"), "paper_ids": [], "concept": None}
    monkeypatch.setenv("HARNESS_RETRIEVAL_MODE", "lexical")
    try:
        runner._retrieve(dict(state))
    except Exception as exc:  # noqa: BLE001
        assert type(exc).__name__ == "NoEvidenceFound"
    else:
        raise AssertionError("lexical retrieval should find no exact word")
    monkeypatch.setenv("HARNESS_RETRIEVAL_MODE", "hybrid")
    result = runner._retrieve(dict(state))
    assert [passage["passage_id"] for passage in result["passages"]] == ["p4-abstract"]
    assert result["retrieval"]["retrieval_mode"] == "hybrid"
    assert result["retrieval"]["queries"][-1]["retrieval"] == "dense"


def test_reading_support_passages_become_retrieval_candidates(tmp_path: Path) -> None:
    runner = Baseline20(corpus(tmp_path), ExtractionEngine(RecordedModel(), hand_written_subschema()), enhanced=True)
    concept = {"label": "Video prediction", "gloss": "Methods that predict video.", "sample_sources": ["p4-abstract"]}
    result = runner._retrieve({"question": "Unfindable zxqvterm", "plan": QueryPlan(operation="research"),
                               "paper_ids": [], "concept": concept})
    assert [passage["passage_id"] for passage in result["passages"]] == ["p4-abstract"]


def test_agent_findings_are_unverified_notes_for_schema_and_answer_only() -> None:
    state = {"question": "Compare SafeRAG and HydraRAG.", "concept": None,
             "agent_findings": [{"statement": "SafeRAG is a benchmark.", "evidence_ids": ["p1-a"]}]}
    assert Baseline20._research_question(state) == "Compare SafeRAG and HydraRAG."
    # Statements only: no ids the answer could cite without a triple behind them.
    assert Baseline20._research_notes(state) == ["SafeRAG is a benchmark."]


def test_reading_support_passages_respect_the_papers_the_question_names(tmp_path: Path) -> None:
    store = corpus(tmp_path)
    (tmp_path / "papers.jsonl").write_text(
        json.dumps({"paper_id": "p4", "title": "Video Prediction Policy", "year": 2025}) + "\n"
        + json.dumps({"paper_id": "p5", "title": "Other", "year": 2025}) + "\n")
    with (tmp_path / "passages.jsonl").open("a") as handle:
        handle.write(json.dumps({"passage_id": "p5-abstract", "paper_id": "p5", "section_path": "Abstract",
                                 "para_idx": 0, "text": "Robot learning with policies."}) + "\n")
    store = CorpusStore(tmp_path)
    runner = Baseline20(store, ExtractionEngine(RecordedModel(), hand_written_subschema()), enhanced=True)
    concept = {"label": "Robot learning", "gloss": "Policies for robots.", "sample_sources": ["p5-abstract"]}
    result = runner._retrieve({"question": "How does it do robot learning?", "plan": QueryPlan(operation="research"),
                               "paper_ids": ["p4"], "concept": concept})
    assert [passage["passage_id"] for passage in result["passages"]] == ["p4-abstract"]


def test_hybrid_mode_ranks_by_the_full_question_when_the_title_was_all_it_had(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_RETRIEVAL_MODE", "hybrid")
    runner = Baseline20(corpus(tmp_path), ExtractionEngine(RecordedModel(), hand_written_subschema()))
    result = runner._retrieve({"question": "What is the Video Prediction Policy?", "plan": QueryPlan(operation="research"),
                               "paper_ids": ["p4"], "concept": None})
    assert [passage["passage_id"] for passage in result["passages"]] == ["p4-abstract"]
