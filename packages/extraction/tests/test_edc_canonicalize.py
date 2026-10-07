"""EDC canonicalization: lexical candidates, the mapping model's choice, paper-subject and direction checks,
kept relations (made-up entities only)."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

import pytest
from harness_extraction import ExtractionEngine, ModelFailure, SchemaInference
from harness_extraction.call_control import call_budget
from harness_extraction.schemas import EDCOpenTriple
from harness_semantic import TOP_CLASS, hand_written_subschema
from rdflib import RDFS

PASSAGE = {"passage_id": "p97-exp-1", "paper_id": "p97", "title": "MethodX for BenchY Tasks",
           "text": "We evaluate MethodX on BenchY. Smith authored this paper."}
INFERRED = SchemaInference(label="x", gloss="x", classes=["Paper", "Dataset"],
                           properties=["usesDataset", "evaluatesOn", "authoredBy", "uses", "outperforms"])


def _open(subject: str, predicate: str, obj: str, source: str = "p97-exp-1") -> dict[str, str]:
    return {"subject": subject, "predicate": predicate, "object": obj, "source_id": source,
            "evidence": f"{subject} {predicate} {obj}"}


class EDCModel:
    """Extraction, definitions and mapping by the prompt's opening words; the mapping reply is a function of
    the propositions the mapping model receives."""

    def __init__(self, triples: list[dict[str, str]],
                 mapper: Callable[[list[dict[str, Any]]], Any] = lambda rows: {"mappings": []},
                 definitions: dict | None = None) -> None:
        self.triples = triples
        self.mapper = mapper
        self.definitions = definitions if definitions is not None else {"definitions": []}
        self.map_calls: list[tuple[str, dict]] = []

    def complete_json(self, system: str, user: str) -> dict:
        if system.startswith("You extract open propositions"):
            return {"triples": self.triples}
        if system.startswith("You define relation"):
            return self.definitions
        if system.startswith("You map open relations"):
            payload = json.loads(user)
            self.map_calls.append((system, payload))
            return self.mapper(payload["propositions"])
        raise AssertionError(system[:60])


def _choose(property_name: str | None, confidence: float = 0.9) -> Callable[[list[dict[str, Any]]], dict]:
    return lambda rows: {"mappings": [{"id": row["id"], "property": property_name, "confidence": confidence}
                                      for row in rows]}


def _run(model: EDCModel, passages: list[dict] | None = None, engine: ExtractionEngine | None = None):
    engine = engine or ExtractionEngine(model, hand_written_subschema())
    result, stages = engine.extract_edc("q", INFERRED, passages or [PASSAGE])
    decisions = next(stage for stage in stages if stage["stage"] == "canonicalize")["decisions"]
    return engine, [(t.subject, t.predicate, t.object) for t in result.triples], decisions, stages


def test_camel_case_and_iri_targets_are_split_into_words() -> None:
    assert ExtractionEngine._norm("usesDataset") == "uses dataset"
    assert ExtractionEngine._norm("hrn:usesDataset") == "uses dataset"
    assert ExtractionEngine._norm("https://harness.local/schema#reportsMetric") == "reports metric"
    assert ExtractionEngine._norm("BERTLarge uses") == "bert large uses"
    # The word-overlap term now counts for multi-word property names: an exact wording scores 1.0.
    assert ExtractionEngine._canonical_property("uses dataset", "", ["uses", "usesDataset"]) == ("usesDataset", 1.0)
    # Function words do not count as overlap: a shared "by" is not a match.
    assert ExtractionEngine._canonical_property("is outperformed by", "", ["authoredBy"])[0] is None


def test_every_proposition_goes_to_the_model_which_may_reject_a_lexical_candidate(monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_MAP_MODEL_MIN", "0.65")
    model = EDCModel([_open("p97", "uses dataset", "BenchY"), _open("p97", "does not use dataset", "BenchZ")],
                     mapper=lambda rows: {"mappings": [{"id": 0, "property": "usesDataset", "confidence": 0.9},
                                                       {"id": 1, "property": None, "confidence": 0.1}]})
    _, triples, decisions, stages = _run(model)
    assert triples == [("p97", "usesDataset", "BenchY")]
    assert [(row["method"], row["selected"]) for row in decisions] == [
        ("ontology_review", "usesDataset"), ("model_rejected", None)]
    assert decisions[1]["candidates"][0] == "usesDataset"      # proposed lexically, rejected by the model
    assert stages[2] == {"stage": "ontology_review", "propositions": 2, "mapped": 1}
    # One mapping call for the batch, also for the exact lexical match; it sees the candidates with
    # domain and range, the paper id and the full catalog, and the threshold from the settings.
    assert len(model.map_calls) == 1
    system, payload = model.map_calls[0]
    first = payload["propositions"][0]
    assert first["paper_id"] == "p97" and first["subject"] == "p97" and first["object"] == "BenchY"
    assert first["candidates"][0] == {"property": "usesDataset", "domain": "Paper", "range": "Dataset"}
    assert {row["property"] for row in payload["allowed_properties"]} == set(INFERRED.properties)
    assert "below 0.65 are not used" in system and "Polarity must match" in system


def test_a_paper_object_never_becomes_the_subject_of_a_paper_property() -> None:
    # A careless model maps "Smith authored p97" onto authoredBy (Paper -> Author).
    model = EDCModel([_open("Smith", "authored", "p97")], mapper=_choose("authoredBy"))
    _, triples, decisions, stages = _run(model)
    assert triples == []                                       # not "p97 authoredBy p97"
    assert decisions[0]["method"] == "inverted_direction" and decisions[0]["rejected"] == "authoredBy"
    assert stages[-1]["triples"][0]["reason"] == "inverted_direction"


def test_an_entity_subject_is_never_rewritten_to_the_paper(monkeypatch) -> None:
    triples_in = [_open("MethodX", "evaluated on", "BenchY"), _open("We", "evaluate on", "BenchZ"),
                  _open("MethodX for BenchY tasks", "evaluated on", "BenchW")]
    _, triples, decisions, _ = _run(EDCModel(triples_in, mapper=_choose("evaluatesOn")))
    # The paper's self-reference and its title become the paper id; the method's fact is rejected.
    assert triples == [("p97", "evaluatesOn", "BenchZ"), ("p97", "evaluatesOn", "BenchW")]
    assert decisions[0]["method"] == "domain_mismatch" and decisions[0]["rejected"] == "evaluatesOn"
    # With kept relations the method's fact survives under its own wording, subject unchanged.
    monkeypatch.setenv("HARNESS_UNMAPPED_RELATIONS", "keep")
    _, kept, decisions, _ = _run(EDCModel(triples_in[:1], mapper=_choose("evaluatesOn")))
    assert kept == [("MethodX", "evaluatedOn", "BenchY")]
    assert decisions[0]["method"] == "kept_new_relation" and decisions[0]["unmapped_because"] == "domain_mismatch"


def test_kept_relations_get_the_generic_domain_and_keep_their_subjects(monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_UNMAPPED_RELATIONS", "keep")
    first = EDCModel([_open("p97", "is limited by", "memory")], mapper=_choose(None))
    engine, triples, _, _ = _run(first)
    assert triples == [("p97", "isLimitedBy", "memory")]
    domain = engine.schema.graph.value(engine.property_names["isLimitedBy"], RDFS.domain)
    assert domain == TOP_CLASS                                 # not Paper, although the first subject was
    # A later batch (another paper) uses the same kept relation for a method: its subject stays.
    later = {"passage_id": "p96-exp-1", "paper_id": "p96", "text": "MethodY is limited by GPU count."}
    engine.model = engine.edc_model = EDCModel([_open("MethodY", "is limited by", "GPU count", "p96-exp-1")],
                                               mapper=_choose(None))
    _, triples, _, _ = _run(engine.edc_model, [later], engine)
    assert triples == [("MethodY", "isLimitedBy", "GPU count")]


def test_one_malformed_definition_keeps_the_others() -> None:
    reply = {"definitions": [{"relation": "uses", "definition": "d1"}, {"relation": "trained on"},
                             "not an object", {"relation": "evaluated on", "definition": "d3"}]}
    engine = ExtractionEngine(EDCModel([], definitions=reply), hand_written_subschema())
    rows = [EDCOpenTriple(subject="a", predicate=p, object="b", evidence_ids=["x"])
            for p in ("uses", "trained on", "evaluated on")]
    assert [(item.relation, item.definition) for item in engine._define(rows)] == [
        ("uses", "d1"), ("evaluated on", "d3")]


def test_string_ids_are_accepted_and_bad_confidences_and_omissions_are_reported() -> None:
    reply = {"mappings": [{"id": "0", "property": "usesDataset", "confidence": 0.9},
                          {"id": 1, "property": "usesDataset", "confidence": 85},
                          {"id": 3, "property": "usesDataset", "confidence": 0.9},     # no such id
                          {"id": 0, "property": None, "confidence": 0}]}               # a second verdict for 0
    model = EDCModel([_open("p97", "uses dataset", "BenchY"), _open("p97", "uses dataset", "BenchZ"),
                      _open("p97", "uses dataset", "BenchW")], mapper=lambda rows: reply)
    _, triples, decisions, stages = _run(model)
    assert triples == [("p97", "usesDataset", "BenchY")]
    assert [row["method"] for row in decisions] == ["ontology_review", "model_invalid", "no_answer"]
    assert [row["reason"] for row in stages[-1]["triples"]] == ["model_invalid", "no_answer"]


def test_lexical_acceptance_is_only_the_fallback_when_the_mapping_call_fails() -> None:
    def fail(rows):
        raise ModelFailure("provider error")

    model = EDCModel([_open("p97", "uses dataset", "BenchY"), _open("MethodX", "evaluated on", "BenchY")],
                     mapper=fail)
    _, triples, decisions, stages = _run(model)
    assert triples == [("p97", "usesDataset", "BenchY")]
    # The checks still apply to the fallback: the method's fact is not moved onto the paper.
    assert [row["method"] for row in decisions] == ["lexical_fallback", "domain_mismatch"]
    assert stages[2]["error"].startswith("ModelFailure")
    # A reply that addresses no proposition counts as a failed call as well.
    _, triples, decisions, _ = _run(EDCModel([_open("p97", "uses dataset", "BenchY")]))
    assert triples == [("p97", "usesDataset", "BenchY")] and decisions[0]["method"] == "lexical_fallback"


def test_a_stopped_run_does_not_fall_back() -> None:
    stopped = {"now": False}

    def stop(rows):
        stopped["now"] = True
        raise ModelFailure("run stopped")

    with call_budget(time.monotonic() + 60, lambda: stopped["now"]), pytest.raises(ModelFailure):
        _run(EDCModel([_open("p97", "uses dataset", "BenchY")], mapper=stop))
