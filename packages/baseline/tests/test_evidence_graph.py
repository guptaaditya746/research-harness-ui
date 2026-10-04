"""A research run keeps everything it extracted (evidence_graph.json) and can keep unmapped relations."""

from __future__ import annotations

import json
from pathlib import Path

from harness_baseline import Baseline20
from harness_baseline.planner import QueryPlan
from harness_corpus.store import CorpusStore
from harness_extraction import ExtractionEngine
from harness_semantic import hand_written_subschema

TEXT = "MethodX is fine-tuned from ModelZ and is trained on the BenchY dataset for claim verification."


class Planner:
    def plan(self, question: str, concept: dict | None = None) -> QueryPlan:
        return QueryPlan(operation="research")


class ScriptedModel:
    """Answers each prompt by its opening words; extraction cites the passages it was given."""

    model = "scripted"

    def complete_json(self, system: str, user: str) -> dict:
        payload = json.loads(user)
        if system.startswith("You design"):
            return {"label": "Data", "gloss": "Datasets", "classes": ["Paper"], "properties": ["usesDataset"],
                    "class_definitions": [], "property_definitions": []}
        if system.startswith("You extract open propositions"):
            source = payload["passages"][0]["source_id"]
            return {"triples": [
                {"subject": "MethodX", "predicate": "is fine-tuned from", "object": "ModelZ",
                 "source_id": source, "evidence": "MethodX is fine-tuned from ModelZ"},
                {"subject": payload["passages"][0]["paper_id"], "predicate": "uses dataset", "object": "BenchY",
                 "source_id": source, "evidence": "trained on the BenchY dataset"},
            ]}
        if system.startswith("You define relation"):
            return {"definitions": []}
        if system.startswith("You map open relations"):
            return {"mappings": []}
        if system.startswith("You review candidate"):
            keep = [row["id"] for row in payload["candidates"] if row["predicate"] == "usesDataset"]
            return {"keep_ids": keep, "drop_reasons": {str(row["id"]): "irrelevant" for row in payload["candidates"]
                                                       if row["id"] not in keep}}
        if system.startswith("You write the final answer"):
            source = payload["triples"][0]["source_id"]
            return {"answer": f"p99 uses BenchY [{source}].", "cited_source_ids": [source]}
        if system.startswith("You check a draft answer"):
            source = payload["triples"][0]["source_id"]
            text = f"p99 uses BenchY [{source}]."
            return {"sentences": [{"text": text, "supported": True, "source_ids": [source], "reason": ""}],
                    "removed_claims": [], "supported": True, "answer": text, "cited_source_ids": [source]}
        if system.startswith("You complete the context"):
            return {"enrichments": []}
        raise AssertionError(f"unexpected prompt: {system[:60]}")


def _store(tmp_path: Path) -> CorpusStore:
    rows = {
        "papers.jsonl": [{"paper_id": "p99", "title": "MethodX for claim verification", "year": 2025}],
        "passages.jsonl": [{"passage_id": "p99-exp-1", "paper_id": "p99", "section_path": "Experiments",
                            "para_idx": 0, "text": TEXT}],
        "facts.jsonl": [], "references.jsonl": [],
    }
    for name, values in rows.items():
        (tmp_path / name).write_text("".join(json.dumps(row) + "\n" for row in values))
    return CorpusStore(tmp_path)


def _run(tmp_path: Path, monkeypatch, mode: str) -> dict:
    monkeypatch.setenv("HARNESS_UNMAPPED_RELATIONS", mode)
    model = ScriptedModel()
    runner = Baseline20(_store(tmp_path), ExtractionEngine(model, hand_written_subschema()), enhanced=True,
                        edc=True, refine=True, refinement_model=model, planner=Planner())
    return runner.run("Which datasets is MethodX trained on, and what is it fine-tuned from?", run_id="r-ev")


def test_evidence_graph_keeps_what_review_dropped_and_kept_relations_join_the_subschema(tmp_path, monkeypatch) -> None:
    result = _run(tmp_path, monkeypatch, "keep")
    run = result["run"]
    assert run.status == "answered", run.reason
    graph = json.loads(result["artifacts"]["evidence_graph.json"])
    rows = {(row["subject"], row["predicate"], row["object"]): row["relevant"] for row in graph["triples"]}
    assert rows == {("MethodX", "isFineTunedFrom", "ModelZ"): False, ("p99", "usesDataset", "BenchY"): True}
    assert graph["counts"]["relevant"] == 1 and graph["counts"]["not_relevant"] == 1
    assert "isFineTunedFrom" in graph["properties"]          # the kept relation is part of the run's subschema
    assert any(row.action == "Keeping unmapped relations" for row in run.trace)
    # The answer still uses only what the review kept.
    assert {triple.p for triple in run.readings[0].triples} == {"usesDataset"}


def test_without_keep_the_unmapped_fact_is_listed_as_dropped(tmp_path, monkeypatch) -> None:
    result = _run(tmp_path, monkeypatch, "drop")
    graph = json.loads(result["artifacts"]["evidence_graph.json"])
    assert [row["predicate"] for row in graph["triples"]] == ["usesDataset"]
    assert graph["counts"]["unmapped_dropped"] == 1
    assert graph["unmapped_dropped"][0]["predicate"] == "is fine-tuned from"
