"""Fact operations added for the question set: authors in several venues, title terms, papers per venue."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from harness_baseline import Baseline20
from harness_baseline.facts import title_has_term, venue_matches
from harness_baseline.planner import QueryPlan
from harness_corpus.store import CorpusStore
from harness_extraction import ExtractionEngine
from harness_semantic import hand_written_subschema
from test_baseline import RecordedModel, StubPlanner

PAPERS = {
    # key: (title, venue as DBLP writes it, year, authors)
    "a1": ("Hallucination Detection in Long Answers", "ACL (1)", 2025, ["Ada Lovelace", "Alan Turing"]),
    "a2": ("Findings on Retrieval", "Findings of ACL", 2025, ["Grace Hopper"]),
    "e1": ("Mitigating Hallucinations with Retrieval", "EMNLP", 2025, ["Ada Lovelace"]),
    "e2": ("Multilingual Safety", "EMNLP", 2025, ["Grace Hopper", "Alan Turing"]),
    "n1": ("Manipulate Me: Hallucination Games", "NAACL", 2024, ["Ada Lovelace"]),
}


def store(tmp_path: Path) -> CorpusStore:
    facts = []
    for key, (title, venue, year, authors) in PAPERS.items():
        facts += [
            {"fact_id": f"{key}-title", "subject": key, "predicate": "title", "object": title, "source_id": f"{key}-metadata"},
            {"fact_id": f"{key}-venue", "subject": key, "predicate": "venue", "object": venue, "source_id": f"{key}-metadata"},
            {"fact_id": f"{key}-year", "subject": key, "predicate": "year", "object": str(year), "source_id": f"{key}-metadata"},
        ]
        facts += [{"fact_id": f"{key}-author-{index}", "subject": key, "predicate": "author", "object": name,
                   "source_id": f"{key}-metadata"} for index, name in enumerate(authors)]
    rows = {
        "papers.jsonl": [{"paper_id": key, "title": value[0], "year": value[2]} for key, value in PAPERS.items()],
        "passages.jsonl": [{"passage_id": f"{key}-abstract", "paper_id": key, "section_path": "Abstract",
                            "para_idx": 0, "text": value[0]} for key, value in PAPERS.items()],
        "facts.jsonl": facts,
    }
    for name, values in rows.items():
        (tmp_path / name).write_text("".join(json.dumps(row) + "\n" for row in values))
    return CorpusStore(tmp_path)


def answer(tmp_path: Path, plan: QueryPlan) -> dict:
    question = "q"
    runner = Baseline20(store(tmp_path), ExtractionEngine(RecordedModel(), hand_written_subschema()),
                        planner=StubPlanner({question: plan}))
    return runner.run(question, run_id="r-facts")


def test_venue_and_title_rules() -> None:
    assert venue_matches("ACL (1)", "2025", "ACL 2025")            # year from the paper, not the venue string
    assert venue_matches("Findings of ACL", "2025", "ACL 2025")    # whole words, as in labelling
    assert not venue_matches("NAACL", "2025", "ACL 2025")
    assert not venue_matches("ACL (1)", "2024", "ACL 2025")
    assert venue_matches("EMNLP 2025", None, "EMNLP 2025")
    assert title_has_term("Mitigating Hallucinations with Retrieval", "hallucination")
    assert not title_has_term("Manipulate Me", "manipulation")


def test_authors_in_venues_intersects_the_venues(tmp_path: Path) -> None:
    result = answer(tmp_path, QueryPlan(operation="authors_in_venues", venues=["ACL 2025", "EMNLP 2025"]))
    run = result["run"]
    assert run.status == "answered"
    reading = run.readings[0]
    authors = {triple.o for triple in reading.triples if triple.p == "authoredBy"}
    # Ada (a1 + e1), Alan (a1 + e2), Grace (a2 Findings of ACL + e2); the NAACL paper is not in scope.
    assert authors == {"Ada Lovelace", "Alan Turing", "Grace Hopper"}
    assert not any("n1" in triple.s for triple in reading.triples)
    assert "3 authors have indexed papers in all of ACL 2025, EMNLP 2025" in reading.answer


def test_papers_with_title_term_honours_the_year(tmp_path: Path) -> None:
    run = answer(tmp_path, QueryPlan(operation="papers_with_title_term", term="hallucination", year=2025))["run"]
    titles = {triple.o for triple in run.readings[0].triples}
    assert titles == {"Hallucination Detection in Long Answers", "Mitigating Hallucinations with Retrieval"}
    everything = answer(tmp_path, QueryPlan(operation="papers_with_title_term", term="hallucination"))["run"]
    assert len(everything.readings[0].triples) == 3                  # the 2024 NAACL paper too


def test_papers_in_venue_counts_and_absence(tmp_path: Path) -> None:
    run = answer(tmp_path, QueryPlan(operation="papers_in_venue", venues=["EMNLP 2025"]))["run"]
    assert run.readings[0].answer.startswith("2 indexed papers were published in EMNLP 2025")
    absent = answer(tmp_path, QueryPlan(operation="papers_in_venue", venues=["COLING 2025"]))["run"]
    assert absent.status == "corpus_absent"


@pytest.mark.parametrize(("plan", "ok"), [
    (QueryPlan(operation="authors_in_venues", venues=["ACL 2025"]), False),
    (QueryPlan(operation="authors_in_venues", venues=["ACL 2025", "EMNLP 2025"]), True),
    (QueryPlan(operation="papers_with_title_term"), False),
    (QueryPlan(operation="papers_with_title_term", term=" 'hallucination' "), True),
    (QueryPlan(operation="papers_in_venue", venues="EMNLP 2025"), True),
])
def test_requirements(plan: QueryPlan, ok: bool) -> None:
    assert plan.requirements_met() is ok


def test_named_papers_share_the_passage_budget(tmp_path: Path) -> None:
    rows = {
        "papers.jsonl": [{"paper_id": "p4", "title": "SafeRAG", "year": 2025},
                         {"paper_id": "p5", "title": "HydraRAG", "year": 2025}],
        "passages.jsonl": [
            {"passage_id": f"{paper}-{index}", "paper_id": paper, "section_path": "Method", "para_idx": index,
             "text": f"Retrieval augmented generation problems and attacks, part {index} of {paper}."}
            for paper in ("p4", "p5") for index in range(8)
        ],
        "facts.jsonl": [],
    }
    for name, values in rows.items():
        (tmp_path / name).write_text("".join(json.dumps(row) + "\n" for row in values))
    seeds = [f"p4-{index}" for index in range(8)]                     # the agent only read SafeRAG
    runner = Baseline20(CorpusStore(tmp_path), ExtractionEngine(RecordedModel(), hand_written_subschema()),
                        enhanced=True, max_passages=8, seed_passage_ids=seeds)
    runner.started = __import__("time").monotonic()
    state = runner._retrieve({"question": "How do SafeRAG and HydraRAG differ in the problems they address in "
                                          "retrieval augmented generation?",
                              "plan": QueryPlan(operation="research"), "paper_ids": ["p4", "p5"], "concept": None})
    papers = [row["paper_id"] for row in state["passages"]]
    assert len(papers) == 8 and papers.count("p5") >= 2              # 8 // (2 * 2) = 2 reserved for HydraRAG


def test_paper_citation_counts_ranks_the_most_cited_paper(tmp_path: Path) -> None:
    corpus_store = store(tmp_path)
    (tmp_path / "references.jsonl").write_text("".join(json.dumps(row) + "\n" for row in [
        {"reference_id": "e1-ref-1", "paper_id": "e1", "title": "Hallucination Detection", "resolved_paper_id": "a1"},
        {"reference_id": "e2-ref-1", "paper_id": "e2", "title": "Hallucination Detection", "resolved_paper_id": "a1"},
        {"reference_id": "e2-ref-2", "paper_id": "e2", "title": "Findings on Retrieval", "resolved_paper_id": "a2"},
    ]))
    corpus_store = CorpusStore(tmp_path)
    runner = Baseline20(corpus_store, ExtractionEngine(RecordedModel(), hand_written_subschema()),
                        planner=StubPlanner({"q": QueryPlan(operation="paper_citation_counts", rank="top")}))
    run = runner.run("q", run_id="r-cited")["run"]
    assert run.readings[0].answer.startswith("Most cited by other indexed papers: Hallucination Detection in Long "
                                             "Answers (2 citing papers)")
    assert {triple.p for triple in run.readings[0].triples} == {"cites"}
