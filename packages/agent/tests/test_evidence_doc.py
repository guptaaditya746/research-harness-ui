"""Evidence documents: every paper screened, every relevant passage extracted, resumable (made-up corpus)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from harness_agent.evidence_doc import DocModels, DocumentBuilder, render_markdown, screening_rows
from harness_corpus.store import CorpusStore

PAPERS = [
    ("p97", "MethodX for claim verification", [
        "MethodX is fine-tuned from ModelZ for claim verification.",
        "MethodX is trained on the BenchY dataset and evaluated with F1.",
    ]),
    ("p98", "A study of poetry generation", ["We generate sonnets with a recurrent network."]),
    ("p99", "Untitled workshop note", ["Qqq rrr sss."]),          # no passage ranks for the question
]


def _store(tmp_path: Path) -> CorpusStore:
    papers = [{"paper_id": pid, "title": title, "year": 2025} for pid, title, _ in PAPERS]
    passages = [{"passage_id": f"{pid}-{n}", "paper_id": pid, "section_path": "Body", "para_idx": n, "text": text}
                for pid, _, texts in PAPERS for n, text in enumerate(texts)]
    tmp_path.mkdir(parents=True, exist_ok=True)
    for name, values in {"papers.jsonl": papers, "passages.jsonl": passages, "facts.jsonl": [],
                         "references.jsonl": []}.items():
        (tmp_path / name).write_text("".join(json.dumps(row) + "\n" for row in values))
    return CorpusStore(tmp_path)


class Model:
    """Readers find MethodX passages relevant; extraction states one fact per passage."""

    model = "stub"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def complete_json(self, system: str, user: str) -> dict:
        payload = json.loads(user)
        self.calls.append(system[:30])
        if system.startswith("You are one of several readers"):
            hits = [row["passage_id"] for row in payload["passages"] if "MethodX" in row["text"]]
            return {"findings": [{"statement": "MethodX is described.", "evidence_ids": hits}] if hits else []}
        if system.startswith("You design"):
            return {"label": "Methods", "gloss": "Methods and data", "classes": ["Paper", "Method"],
                    "properties": ["usesDataset"], "class_definitions": [], "property_definitions": []}
        if system.startswith("You extract open propositions"):
            row = payload["passages"][0]
            if "fine-tuned" in row["text"]:
                return {"triples": [{"subject": "MethodX", "predicate": "is fine-tuned from", "object": "ModelZ",
                                     "source_id": row["source_id"], "evidence": "MethodX is fine-tuned from ModelZ"}]}
            return {"triples": [{"subject": row["paper_id"], "predicate": "uses dataset", "object": "BenchY",
                                 "source_id": row["source_id"], "evidence": "trained on the BenchY dataset"}]}
        if system.startswith("You define relation"):
            return {"definitions": []}
        if system.startswith("You map open relations"):
            return {"mappings": []}
        if system.startswith("You are the planning agent"):
            return {"concepts": [
                {"label": "Most used method", "gloss": "Count papers using it.", "classes": ["Method"],
                 "search_terms": ["MethodX"]},
                {"label": "Best scoring method", "gloss": "Highest F1.", "classes": ["Metric"],
                 "search_terms": ["F1"]},
                {"label": "Most extended method", "gloss": "Fine-tuned most often.", "classes": ["Method"],
                 "search_terms": ["fine-tuned"]},
            ]}
        raise AssertionError(f"unexpected prompt: {system[:80]}")


class NoCalls:
    model = "none"

    def complete_json(self, system: str, user: str) -> dict:
        raise AssertionError("a resumed build must not call a model")


@pytest.fixture(autouse=True)
def _settings(monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_UNMAPPED_RELATIONS", "keep")
    monkeypatch.setenv("HARNESS_EDC_BATCH_PASSAGES", "1")
    monkeypatch.setenv("HARNESS_DOC_CONCURRENCY", "2")
    monkeypatch.setenv("HARNESS_HIDE_UNGROUNDED_READINGS", "false")
    monkeypatch.setenv("HARNESS_DOC_MAX_PAPERS", "0")


def _models(model) -> DocModels:
    return DocModels(reader=model, schema=model, extraction=model, edc=model, agent=model)


def test_every_paper_is_screened_and_every_relevant_passage_extracted(tmp_path) -> None:
    corpus = _store(tmp_path / "corpus")
    folder = DocumentBuilder(corpus, _models(Model()), tmp_path / "docs", log=lambda _: None).build(
        {"id": "x1", "category": "corpus_wide", "question": "Which methods use which datasets for claim verification?"})
    doc = json.loads((folder / "doc.json").read_text())
    [reading] = doc["interpretations"]
    coverage = {row["paper_id"]: row for row in reading["coverage"]}
    assert set(coverage) == {"p97", "p98", "p99"}                       # exhaustive: every paper has a row
    assert coverage["p97"]["status"] == "relevant" and coverage["p97"]["triples"] == 2
    assert coverage["p98"]["status"] == "not_relevant"
    assert coverage["p99"]["selection"] == "first_passages"             # read although nothing ranked
    assert {(row["s"], row["p"], row["o"]) for row in reading["triples"]} == {
        ("MethodX", "isFineTunedFrom", "ModelZ"), ("p97", "usesDataset", "BenchY")}
    assert "isFineTunedFrom" in reading["schema"]["properties"]
    assert reading["schema"]["kept_from_extraction"] == {"isFineTunedFrom": "is fine-tuned from"}
    assert all(claim["tripleIds"] for claim in reading["claims"])
    assert reading["answer_key"]["status"] == "pending_review"
    assert (folder / "i1.subschema.ttl").read_text().count("isFineTunedFrom") >= 1
    assert "| t00001 |" in (folder / "doc.md").read_text()


def test_an_interrupted_build_resumes_without_model_calls(tmp_path) -> None:
    corpus = _store(tmp_path / "corpus")
    row = {"id": "x2", "category": "corpus_wide", "question": "Which methods use which datasets for claim verification?"}
    first = DocumentBuilder(corpus, _models(Model()), tmp_path / "docs", log=lambda _: None).build(row)
    before = json.loads((first / "doc.json").read_text())
    again = DocumentBuilder(corpus, _models(NoCalls()), tmp_path / "docs", log=lambda _: None).build(row)
    after = json.loads((again / "doc.json").read_text())
    assert after["interpretations"][0]["triples"] == before["interpretations"][0]["triples"]
    assert after["interpretations"][0]["schema"]["properties"] == before["interpretations"][0]["schema"]["properties"]


def test_ambiguous_questions_get_one_section_per_agent_interpretation(tmp_path) -> None:
    corpus = _store(tmp_path / "corpus")
    folder = DocumentBuilder(corpus, _models(Model()), tmp_path / "docs", log=lambda _: None).build(
        {"id": "x3", "category": "ambiguous", "question": "Which method is the best for claim verification?"})
    doc = json.loads((folder / "doc.json").read_text())
    # The agent's readings, ranked by corpus support as in an agentic run.
    assert {item["label"] for item in doc["interpretations"]} == {
        "Most used method", "Best scoring method", "Most extended method"}
    assert [item["id"] for item in doc["interpretations"]] == ["i1", "i2", "i3"]
    assert all(item["source"] == "agent" and len(item["coverage"]) == 3 for item in doc["interpretations"])
    assert "## i1." in render_markdown(doc)


def test_screening_falls_back_to_reading_order(tmp_path) -> None:
    corpus = _store(tmp_path / "corpus")
    rows, selection = screening_rows(corpus, "claim verification", "p99", 4)
    assert selection == "first_passages" and [row["passage_id"] for row in rows] == ["p99-0"]


def test_progress_is_logged_written_and_reported_by_status(tmp_path, capsys) -> None:
    from harness_agent.evidence_doc import main, print_status

    corpus = _store(tmp_path / "corpus")
    lines: list[str] = []
    folder = DocumentBuilder(corpus, _models(Model()), tmp_path / "docs", log=lines.append, bar=False).build(
        {"id": "x4", "category": "corpus_wide", "question": "Which methods use which datasets for claim verification?"})
    text = "\n".join(lines)
    assert "i1: screening: 3 papers, 3 to go" in text
    assert "i1: screening 3/3 · relevant 1" in text                  # a line at the end of the stage
    assert "i1: extraction (1 papers): 2 passages, 2 to go" in text
    assert "i1: extraction (1 papers) done in" in text and "triples 2" in text
    state = json.loads((folder / "progress.json").read_text())
    # Stub models emit no model events (the real client does); the counters are there either way.
    assert state["stage"] == "done" and set(state["model_calls"]) == {"finished", "failed", "in_flight"}
    assert state["counts"] == {"triples": 2} and state["done"] == state["total"] == 2
    print_status(tmp_path / "docs")
    assert capsys.readouterr().out.startswith("x4         done")
    # A resumed build says what it skipped.
    lines.clear()
    DocumentBuilder(corpus, _models(NoCalls()), tmp_path / "docs", log=lines.append, bar=False).build(
        {"id": "x4", "category": "corpus_wide", "question": "Which methods use which datasets for claim verification?"})
    assert any("3 already done (resumed)" in line for line in lines)
    assert main(["--status", "--out", str(tmp_path / "docs")]) == 0


def test_model_events_count_calls_and_a_pending_call_gets_a_heartbeat(tmp_path) -> None:
    import time

    from harness_agent.evidence_doc import Progress

    lines: list[str] = []
    progress = Progress(tmp_path, "x5", bar=False, log=lines.append, heartbeat_s=0.05)
    progress.phase("i1", "schema inference")
    progress.on_model_event("Model call started", "heavy-model · schema inference")
    time.sleep(0.2)
    progress.on_model_event("Model call finished", "heavy-model · schema inference · 3.1s")
    progress.on_model_event("Model call failed", "heavy-model · timed out")
    progress.close()
    assert any("waiting on 1 model call(s): heavy-model · schema inference" in line for line in lines)
    assert any("model call failed: heavy-model · timed out" in line for line in lines)
    assert json.loads((tmp_path / "progress.json").read_text())["model_calls"] == {
        "finished": 1, "failed": 1, "in_flight": 0}
