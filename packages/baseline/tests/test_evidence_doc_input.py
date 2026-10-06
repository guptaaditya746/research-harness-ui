"""The evidence document D as a run's input: loading it, and a Pipeline run that picks from it (made-up corpus)."""

import json
from pathlib import Path

import pytest
from harness_baseline import Baseline20
from harness_baseline import evidence_doc as docs
from harness_corpus.store import CorpusStore
from harness_extraction import ExtractionEngine
from harness_semantic import hand_written_subschema

QUESTION = "Which benchmarks does MethodX evaluate on?"


def corpus(tmp_path: Path) -> CorpusStore:
    rows = {
        "papers.jsonl": [{"paper_id": "p97", "title": "MethodX for claim verification", "year": 2025}],
        "passages.jsonl": [
            {"passage_id": f"p97-{n}", "paper_id": "p97", "section_path": "Body", "para_idx": n, "text": text}
            for n, text in enumerate(["MethodX is evaluated on the BenchY benchmark.",
                                      "MethodX is trained with the AdamW optimizer."])],
        "facts.jsonl": [], "references.jsonl": [],
    }
    tmp_path.mkdir(parents=True, exist_ok=True)
    for name, values in rows.items():
        (tmp_path / name).write_text("".join(json.dumps(row) + "\n" for row in values))
    return CorpusStore(tmp_path)


def write_doc(root: Path, readings: int = 1) -> Path:
    folder = root / "q90"
    folder.mkdir(parents=True)
    triples = [
        {"id": "t00001", "s": "p97", "p": "evaluatesOn", "o": "BenchY", "paper_id": "p97",
         "sources": [{"source_id": "p97-0", "evidence": "evaluated on the BenchY benchmark"}]},
        {"id": "t00002", "s": "p97", "p": "usesOptimizer", "o": "AdamW", "paper_id": "p97",
         "sources": [{"source_id": "p97-1", "evidence": "trained with the AdamW optimizer"}]},
        {"id": "t00003", "s": "p97", "p": "evaluatesOn", "o": "BenchZ", "paper_id": "p97",
         "sources": [{"source_id": "p55-9", "evidence": "not in this corpus"}]},
    ]
    doc = {"question": {"id": "q90", "question": "Which benchmarks does MethodX evaluate on ?"},
           "interpretations": [{"id": f"i{n}", "label": f"Reading {n}", "gloss": f"gloss {n}", "triples": triples}
                               for n in range(1, readings + 1)]}
    (folder / "doc.json").write_text(json.dumps(doc))
    return folder / "doc.json"


class PickingModel:
    """Evidence review keeps evaluatesOn triples; composition cites their passage."""

    model = "stub"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete_json(self, system: str, user: str) -> dict:
        self.prompts.append(system)
        payload = json.loads(user)
        if "keep_ids" in system:
            return {"keep_ids": [c["id"] for c in payload["candidates"] if c["predicate"] == "evaluatesOn"],
                    "drop_reasons": {}}
        if "answer" in system and "triples" in payload:
            sources = sorted({t["source_id"] for t in payload["triples"]})
            return {"sparql": "SELECT ?s ?o WHERE { ?s ?p ?o . }",
                    "answer": f"MethodX is evaluated on BenchY [{sources[0]}].", "cited_source_ids": sources}
        raise AssertionError(f"unexpected model call: {system[:80]}")


class NoPlanner:
    def plan(self, question: str, concept: dict | None = None):
        raise AssertionError("with D as input the question is not planned (nothing is retrieved)")


def test_the_document_is_found_by_its_question_and_triples_need_a_corpus_passage(tmp_path: Path) -> None:
    store = corpus(tmp_path / "corpus")
    path = write_doc(tmp_path / "docs", readings=2)
    assert docs.find(tmp_path / "docs", QUESTION) == path        # spacing and punctuation aside
    assert docs.find(tmp_path / "docs", "Another question?") is None
    doc = docs.load(tmp_path / "docs", QUESTION, store.by_passage)
    assert doc.question_id == "q90" and doc.left_out == 2 and len(doc.triples) == 4
    assert len(doc.extracted()) == 2                                 # the same fact under two readings once
    assert len(doc.for_reading("i2").triples) == 2 and doc.for_reading("nope") is doc
    assert doc.subset({"t00001"}).summary()["chosen_by_agent"] == 2
    with pytest.raises(docs.EvidenceDocMissing):
        docs.load(tmp_path / "docs", "Another question?", store.by_passage)


def test_a_pipeline_run_picks_from_the_document_instead_of_searching(tmp_path: Path) -> None:
    store = corpus(tmp_path / "corpus")
    write_doc(tmp_path / "docs")
    model = PickingModel()
    runner = Baseline20(store, ExtractionEngine(model, hand_written_subschema()), planner=NoPlanner())
    runner.evidence_doc = docs.load(tmp_path / "docs", QUESTION, store.by_passage)
    run = runner.run(QUESTION, run_id="r-doc")["run"]
    assert run.status == "answered", run.reason
    assert run.run_input == "evidence_doc" and run.evidence_doc["question_id"] == "q90"
    assert [(t.s, t.p, t.o) for t in run.readings[0].triples] == [("p97", "evaluatesOn", "BenchY")]
    checks = {check.name: check.status for check in run.checks}
    assert checks["retrieval"] == "skipped" and checks["evidence review"] == "pass"
    assert sum("keep_ids" in prompt for prompt in model.prompts) == 1


def test_nothing_picked_is_no_results(tmp_path: Path) -> None:
    store = corpus(tmp_path / "corpus")
    write_doc(tmp_path / "docs")

    class KeepsNothing(PickingModel):
        def complete_json(self, system: str, user: str) -> dict:
            return {"keep_ids": [], "drop_reasons": {}}

    runner = Baseline20(store, ExtractionEngine(KeepsNothing(), hand_written_subschema()), planner=NoPlanner())
    runner.evidence_doc = docs.load(tmp_path / "docs", QUESTION, store.by_passage)
    run = runner.run(QUESTION, run_id="r-none")["run"]
    assert run.status == "incomplete" and run.reason == "no_results"
    assert run.run_input == "evidence_doc"
