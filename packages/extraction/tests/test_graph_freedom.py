"""Settings that let extraction keep more of what it reads (made-up entities only)."""

from __future__ import annotations

import json

from harness_extraction import ExtractionEngine, SchemaInference, relation_name
from harness_prompts import render
from harness_semantic import TOP_ATTRIBUTE, TOP_RELATION, hand_written_subschema
from rdflib import RDFS

PASSAGE = {"passage_id": "p99-exp-1", "paper_id": "p99",
           "text": "MethodX is fine-tuned from ModelZ and reaches 71.2 F1 on BenchY."}
OPEN = {"triples": [
    {"subject": "MethodX", "predicate": "is fine-tuned from", "object": "ModelZ",
     "source_id": "p99-exp-1", "evidence": "MethodX is fine-tuned from ModelZ"},
    {"subject": "p99", "predicate": "reaches F1 of", "object": "71.2 on BenchY",
     "source_id": "p99-exp-1", "evidence": "reaches 71.2 F1 on BenchY"},
]}


class OpenModel:
    """Open extraction, definitions, then a mapping model that maps nothing."""

    def __init__(self) -> None:
        self.users: list[dict] = []
        self.systems: list[str] = []
        self.replies = iter([OPEN, {"definitions": [
            {"relation": "is fine-tuned from", "definition": "The method is trained further from the model."}]},
            {"mappings": []}])

    def complete_json(self, system: str, user: str) -> dict:
        self.systems.append(system)
        self.users.append(json.loads(user))
        return next(self.replies)


def _run(monkeypatch, mode: str):
    monkeypatch.setenv("HARNESS_UNMAPPED_RELATIONS", mode)
    engine = ExtractionEngine(OpenModel(), hand_written_subschema())
    inferred = SchemaInference(label="Data", gloss="Datasets", classes=["Paper"], properties=["usesDataset"])
    result, stages = engine.extract_edc("Which datasets do papers use?", inferred, [PASSAGE])
    return engine, result, stages


def test_unmapped_facts_are_dropped_by_default_and_reported(monkeypatch) -> None:
    engine, result, stages = _run(monkeypatch, "drop")
    assert result.triples == [] and engine.kept_relations == {}
    assert stages[-1]["stage"] == "unmapped_dropped" and stages[-1]["count"] == 2


def test_keep_turns_an_unmapped_relation_into_a_typed_sub_property(monkeypatch) -> None:
    engine, result, stages = _run(monkeypatch, "keep")
    assert [(t.subject, t.predicate, t.object) for t in result.triples] == [
        ("MethodX", "isFineTunedFrom", "ModelZ"), ("p99", "reachesF1Of", "71.2 on BenchY")]
    assert engine.kept_relations == {"isFineTunedFrom": "is fine-tuned from", "reachesF1Of": "reaches F1 of"}
    graph = engine.schema.graph
    fine_tuned = engine.property_names["isFineTunedFrom"]
    reaches = engine.property_names["reachesF1Of"]
    # Entity-to-entity relation: object property under researchRelation, with the define step's definition.
    assert graph.value(fine_tuned, RDFS.subPropertyOf) == TOP_RELATION
    assert "trained further" in str(graph.value(fine_tuned, RDFS.comment))
    # A value about the paper: datatype property under researchAttribute, domain Paper.
    assert graph.value(reaches, RDFS.subPropertyOf) == TOP_ATTRIBUTE
    assert str(graph.value(reaches, RDFS.domain)).endswith("#Paper")
    assert all(stage["stage"] != "unmapped_dropped" for stage in stages)
    decisions = next(stage for stage in stages if stage["stage"] == "canonicalize")["decisions"]
    assert {row["method"] for row in decisions} == {"kept_new_relation"}


def test_mapping_thresholds_are_settings(monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_MAP_LEXICAL_MIN", "0.99")
    assert ExtractionEngine._canonical_property("uses dataset", "", ["usesDataset"])[0] is None
    monkeypatch.setenv("HARNESS_MAP_LEXICAL_MIN", "0.2")
    assert ExtractionEngine._canonical_property("uses dataset", "", ["usesDataset"])[0] == "usesDataset"


def test_values_and_names_get_the_right_range() -> None:
    from harness_extraction.engine import _literal_value

    assert _literal_value("71.2 on BenchY") and _literal_value("~3% fewer errors")
    assert not _literal_value("GPT-4") and not _literal_value("Llama 3") and not _literal_value("BenchY")
    assert _literal_value("a long phrase that describes a limitation in many words")


def test_relation_names_are_lower_camel_case() -> None:
    assert relation_name("is evaluated on") == "isEvaluatedOn"
    assert relation_name("Outperforms") == "outperforms"
    assert relation_name("3 times faster than") is None
    assert relation_name("—") is None


def test_broad_scope_and_dense_linking_change_only_the_prompts_that_use_them() -> None:
    focused = render("extraction/schema_inference", {"question": "q"})
    broad = render("extraction/schema_inference", {"question": "q"},
                   schema_scope="broad", graph_linking="dense")
    assert "Keep the subschema small" in focused.system and "Keep the subschema small" not in broad.system
    assert "connected graph" in broad.system and "connected graph" not in focused.system
    plain = render("extraction/edc_extract", {"question": "q"}, prompt_set="v2")
    linked = render("extraction/edc_extract", {"question": "q"}, prompt_set="v2", graph_linking="dense")
    assert "Link the facts" in linked.system and "Link the facts" not in plain.system


def test_the_engine_passes_the_shape_settings_to_the_prompts(monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_GRAPH_LINKING", "dense")
    monkeypatch.setenv("HARNESS_SCHEMA_SCOPE", "broad")
    model = OpenModel()
    engine = ExtractionEngine(model, hand_written_subschema())
    engine.extract_edc("q", SchemaInference(label="x", gloss="x", classes=["Paper"], properties=["usesDataset"]),
                       [PASSAGE])
    assert "Link the facts" in model.systems[0] and "Be exhaustive" in model.systems[0]
