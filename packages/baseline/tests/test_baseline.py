import json
from pathlib import Path

from harness_baseline import Baseline20
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
    runner = Baseline20(
        corpus(tmp_path), ExtractionEngine(RecordedModel(), hand_written_subschema())
    )
    result = runner.run("Which paper introduced Video Prediction Policy?", run_id="r-live")

    assert result["run"].status == "answered"
    assert [row.action for row in result["run"].trace] == [
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


def test_baseline_no_hits_is_a_completed_semantic_no_evidence_run(tmp_path: Path) -> None:
    runner = Baseline20(
        corpus(tmp_path), ExtractionEngine(RecordedModel(), hand_written_subschema())
    )
    result = runner.run("Unfindable zxqvterm", run_id="r-failed")

    assert result["run"].status == "no_evidence_found"
    assert "No evidence was found" in (result["run"].reason or "")
    assert result["run"].search_coverage["coverage_complete"] is False
    assert set(result["artifacts"]) | {"run.json"} == set(required_artifacts("baseline"))


def test_enhanced_baseline_seeds_structural_facts_and_batches_extraction(tmp_path: Path) -> None:
    runner = Baseline20(
        corpus(tmp_path), ExtractionEngine(RecordedModel(), hand_written_subschema()), enhanced=True
    )
    result = runner.run("Which paper introduced Video Prediction Policy?", run_id="r-enhanced")

    assert result["run"].path == "baseline-enhanced"
    assert "retrieval.json" in result["artifacts"]
    assert "extraction_batches.json" in result["artifacts"]
    assert "corpus-seed" in result["artifacts"]["graph.trig"].decode()
    assert set(result["artifacts"]) | {"run.json"} == set(required_artifacts("baseline-enhanced"))


def test_missing_incoming_citations_do_not_return_unrelated_triples(tmp_path: Path) -> None:
    runner = Baseline20(
        corpus(tmp_path), ExtractionEngine(RecordedModel(), hand_written_subschema())
    )
    result = runner.run("Which papers cite Video Prediction Policy?", run_id="r-citations")

    assert result["run"].status == "corpus_absent"
    assert result["run"].evidence_surface == "citation_index"
    assert result["run"].search_coverage["exhaustive"] is True
    assert not result["run"].readings
    assert "No in-corpus evidence found" in result["artifacts"]["answer.md"].decode()


def test_citation_absence_survives_question_paraphrase(tmp_path: Path) -> None:
    runner = Baseline20(corpus(tmp_path), ExtractionEngine(RecordedModel(), hand_written_subschema()))
    run = runner.run("Which papers in this corpus cite Video Prediction Policy?")["run"]
    assert run.status == "corpus_absent"
    assert run.search_coverage["matches"] == 0


def test_preprint_and_journal_version_count_as_one_work(tmp_path: Path) -> None:
    store = corpus(tmp_path)
    store.by_paper["p5"] = {"paper_id": "p5", "title": "Video Prediction Policy", "year": 2025}
    store.by_paper["p6"] = {"paper_id": "p6", "title": "A Different Robot Policy", "year": 2026}
    runner = Baseline20(store, ExtractionEngine(RecordedModel(), hand_written_subschema()))
    rows = [{"subject": paper_id} for paper_id in ("p4", "p5", "p6")]

    assert [row["subject"] for row in runner._distinct_works(rows)] == ["p4", "p6"]


def test_malformed_model_query_uses_verified_triple_predicate(tmp_path: Path) -> None:
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

    assert "<https://harness.local/schema#reportsMetric>" in answer.sparql
    assert "41.5%" in answer.answer
    assert any(row.action == "Repairing graph query" for row in runner.trace)


def test_year_and_venue_use_verified_corpus_facts(tmp_path: Path) -> None:
    store = corpus(tmp_path)
    store.fact_rows.extend([
        {"fact_id": "p4-year", "subject": "p4", "predicate": "year", "object": "2025", "source_id": "p4-metadata"},
        {"fact_id": "p4-venue", "subject": "p4", "predicate": "venue", "object": "Advanced Robotics", "source_id": "p4-metadata"},
    ])
    runner = Baseline20(store, ExtractionEngine(RecordedModel(), hand_written_subschema()))

    year = runner.run("In what year was Video Prediction Policy published?")["run"]
    venue = runner.run("In which venue was Video Prediction Policy published?")["run"]

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
    runner = Baseline20(store, ExtractionEngine(RecordedModel(), hand_written_subschema()))

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


def test_conflicting_benchmark_percentages_are_both_reported(tmp_path: Path) -> None:
    class FailModel:
        def complete_json(self, system: str, user: str) -> dict:
            raise AssertionError("the two extracted claims already answer this question")

    runner = Baseline20(corpus(tmp_path), ExtractionEngine(FailModel(), hand_written_subschema()))
    runner.started = __import__("time").monotonic()
    state = {
        "question": "What improvement did Video Prediction Policy report on CALVIN ABC→D?",
        "extracted": [
            ExtractedTriple(subject="p4", predicate="reportsMetric", object="18.6% relative improvement on CALVIN ABC-D generalization benchmark", source_id="p4-abstract", evidence="18.6% relative improvement"),
            ExtractedTriple(subject="p4", predicate="reportsMetric", object="41.5% improvement on CALVIN ABC→D benchmark", source_id="p4-results", evidence="41.5% improvement"),
                ExtractedTriple(subject="p4", predicate="reportsMetric", object="10% of CALVIN ABC training data", source_id="p4-training", evidence="using only 10% of annotated data"),
                ExtractedTriple(subject="p4", predicate="reportsMetric", object="31.6% improvement in real-world success rate", source_id="p4-real", evidence="31.6% improvement in real-world success rate"),
        ],
    }

    result = runner._query_answer(state)

    assert "18.6%" in result["answer"].answer
    assert "41.5%" in result["answer"].answer
    assert "10%" not in result["answer"].answer
    assert "31.6%" not in result["answer"].answer
    assert result["answer"].cited_source_ids == ["p4-abstract", "p4-results"]
    assert len(result["extracted"]) == 2
