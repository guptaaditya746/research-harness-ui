"""Phase 4: OWL 2 terms with parents and origins, the DBLP backbone, and a query-driven export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from harness_baseline import Baseline20
from harness_corpus.store import CorpusStore
from harness_extraction import ExtractionEngine
from harness_semantic import hand_written_subschema
from rdflib import OWL, RDF, RDFS, Graph, URIRef
from test_baseline import StubPlanner, plan

HRN = "https://harness.local/schema#"
QUESTION = "Which tasks do the robot learning papers address?"


def two_papers(tmp_path: Path) -> CorpusStore:
    rows = {
        "papers.jsonl": [{"paper_id": "p4", "title": "Video Prediction Policy", "year": 2025},
                         {"paper_id": "p5", "title": "World Models for Robots", "year": 2025},
                         {"paper_id": "p9", "title": "Unrelated", "year": 2020}],
        "passages.jsonl": [
            {"passage_id": "p4-abstract", "paper_id": "p4", "section_path": "Abstract", "para_idx": 0,
             "text": "We propose the Video Prediction Policy for robot learning papers and tasks."},
            {"passage_id": "p5-abstract", "paper_id": "p5", "section_path": "Abstract", "para_idx": 0,
             "text": "World models address robot learning tasks from video, as in the papers above."},
        ],
        "facts.jsonl": [
            {"fact_id": "p4-author-0", "subject": "p4", "predicate": "author", "object": "Ada Lovelace",
             "source_id": "p4-metadata"},
            {"fact_id": "p5-author-0", "subject": "p5", "predicate": "author", "object": "Ada Lovelace",
             "source_id": "p5-metadata"},
            {"fact_id": "p4-venue", "subject": "p4", "predicate": "venue", "object": "ACL 2025",
             "source_id": "p4-metadata"},
            {"fact_id": "p5-venue", "subject": "p5", "predicate": "venue", "object": "ACL 2025",
             "source_id": "p5-metadata"},
            {"fact_id": "p4-year", "subject": "p4", "predicate": "year", "object": "2025", "source_id": "p4-metadata"},
            {"fact_id": "p5-ref-1-cites", "subject": "p5", "predicate": "cites", "object": "p4",
             "source_id": "p5-ref-1", "in_corpus": True},
            {"fact_id": "p9-author-0", "subject": "p9", "predicate": "author", "object": "Someone Else",
             "source_id": "p9-metadata"},
        ],
        "references.jsonl": [{"reference_id": "p5-ref-1", "paper_id": "p5", "title": "Video Prediction Policy",
                              "resolved_paper_id": "p4"}],
    }
    for name, values in rows.items():
        (tmp_path / name).write_text("".join(json.dumps(row) + "\n" for row in values))
    return CorpusStore(tmp_path)


class SchemaModel:
    def complete_json(self, system: str, user: str) -> dict:
        self.payload = json.loads(user)
        return {
            "label": "Tasks", "gloss": "Which task each paper addresses",
            "classes": ["Paper", "Task"], "properties": ["addresses", "robotCount"],
            "class_definitions": [{"name": "Task", "definition": "A research problem.", "parent": "ResearchEntity"}],
            "property_definitions": [
                {"name": "addresses", "definition": "The paper works on the task.", "domain": "Paper",
                 "range": "Task", "parent": "researchRelation"},
                # A datatype property with an object parent: the parent is replaced by the generic one.
                {"name": "robotCount", "definition": "Number of robots.", "domain": "Paper",
                 "range": "xsd:string", "parent": "usesDataset"},
            ],
        }


class AnswerModel:
    def __init__(self) -> None:
        self.replies = iter([
            {"triples": [
                {"subject": "p4", "predicate": "addresses", "object": "robot learning",
                 "source_id": "p4-abstract", "evidence": "for robot learning"},
                {"subject": "p5", "predicate": "addresses", "object": "robot learning",
                 "source_id": "p5-abstract", "evidence": "address robot learning tasks"},
            ]},
            {"sparql": f"PREFIX hrn: <{HRN}>\nSELECT ?p ?t WHERE {{ ?p hrn:addresses ?t . }}",
             "answer": "Both address robot learning [p4-abstract] [p5-abstract].",
             "cited_source_ids": ["p4-abstract", "p5-abstract"]},
        ])

    def complete_json(self, system: str, user: str) -> dict:
        return next(self.replies)


def run(tmp_path: Path) -> tuple[dict, SchemaModel]:
    schema_model = SchemaModel()
    runner = Baseline20(two_papers(tmp_path),
                        ExtractionEngine(AnswerModel(), hand_written_subschema(), schema_model=schema_model),
                        planner=StubPlanner({QUESTION: plan("research")}))
    return runner.run(QUESTION, run_id="r-backbone"), schema_model


def test_invented_terms_are_owl2_with_parents_and_marked_by_origin(tmp_path: Path) -> None:
    result, schema_model = run(tmp_path)
    reading = result["run"].readings[0]
    assert result["run"].status == "answered"
    assert "researchRelation (object)" in schema_model.payload["parent_properties"]
    assert "dblp:authoredBy (object)" in schema_model.payload["parent_properties"]
    terms = {term.name: term for term in reading.schema_terms}
    assert (terms["hrn:addresses"].kind, terms["hrn:addresses"].origin, terms["hrn:addresses"].parent) == (
        "object", "invented", "hrn:researchRelation")
    assert (terms["hrn:robotCount"].kind, terms["hrn:robotCount"].parent) == ("datatype", "hrn:researchAttribute")
    assert (terms["hrn:Task"].kind, terms["hrn:Task"].origin, terms["hrn:Task"].parent) == (
        "class", "invented", "hrn:ResearchEntity")
    assert terms["hrn:Paper"].origin == "core"
    assert terms["dblp:authoredBy"].origin == "dblp"            # backbone predicate, shown by its DBLP name
    schema = Graph().parse(data=result["artifacts"]["subschema.ttl"].decode(), format="turtle")
    addresses = URIRef(HRN + "addresses")
    assert (addresses, RDF.type, OWL.ObjectProperty) in schema
    assert (addresses, RDFS.subPropertyOf, URIRef(HRN + "researchRelation")) in schema
    assert (URIRef(HRN + "Task"), RDF.type, OWL.Class) in schema
    # Harness classes are OWL 2 classes (the DBLP seed file itself also says rdfs:Class; it is kept as is).
    assert not [c for c in schema.subjects(RDF.type, RDFS.Class) if str(c).startswith(HRN)]
    # Only the question's terms: an unrelated DBLP property is not exported.
    assert not list(schema.triples((URIRef("https://dblp.org/rdf/schema#coEditorWith"), None, None)))


def test_backbone_connects_the_papers_but_stays_out_of_the_scored_triples(tmp_path: Path) -> None:
    result, _ = run(tmp_path)
    reading = result["run"].readings[0]
    assert [triple.p for triple in reading.triples] == ["addresses", "addresses"]
    edges = {(edge.label, edge.origin) for edge in reading.graph.edges if edge.kind == "fact"}
    assert {("addresses", "invented"), ("authoredBy", "dblp"), ("publishedInStream", "dblp"),
            ("cites", "core")} <= edges
    assert not any(edge.label in {"year", "yearOfPublication"} for edge in reading.graph.edges)
    venue = [node for node in reading.graph.nodes if node.label == "ACL 2025"]
    assert len(venue) == 1 and venue[0].kind == "entity"                # one shared venue node
    export = result["artifacts"]["triples.ttl"].decode()
    assert "Someone%20Else" not in export and "Ada%20Lovelace" in export    # only the answer's papers


def test_backbone_and_new_classes_can_be_switched_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HARNESS_DBLP_BACKBONE", "false")
    monkeypatch.setenv("HARNESS_SCHEMA_NEW_CLASSES", "off")
    result, _ = run(tmp_path)
    reading = result["run"].readings[0]
    assert not any(edge.label in {"authoredBy", "publishedInStream"} for edge in reading.graph.edges)
    assert "Task" not in reading.classes and "ResearchEntity" in reading.classes
    assert "Someone%20Else" in result["artifacts"]["triples.ttl"].decode()   # the old corpus-wide export


def test_graph_edge_ids_are_unique_with_the_backbone(tmp_path: Path) -> None:
    result, _ = run(tmp_path)
    ids = [edge.id for edge in result["run"].readings[0].graph.edges]
    assert len(ids) == len(set(ids))                                     # cytoscape refuses duplicates
    assert any(edge_id.startswith("b-") for edge_id in ids)


def test_generic_parents_are_not_predicates_and_parent_names_are_normalised(tmp_path: Path) -> None:
    from harness_semantic import property_kind

    class ParentsAsProperties(SchemaModel):
        def complete_json(self, system: str, user: str) -> dict:
            reply = super().complete_json(system, user)
            reply["properties"] = [*reply["properties"], "researchAttribute"]
            reply["property_definitions"][0]["parent"] = "hrn:usesDataset (object)"
            return reply

    engine = ExtractionEngine(AnswerModel(), hand_written_subschema(), schema_model=ParentsAsProperties())
    inferred = engine.infer_schema(QUESTION, [{"passage_id": "p4-abstract", "paper_id": "p4", "text": "robot"}])
    assert "researchAttribute" not in inferred.properties
    graph = engine.schema.graph
    attribute = URIRef(HRN + "researchAttribute")
    assert property_kind(graph, attribute) == "datatype" and (attribute, RDF.type, OWL.ObjectProperty) not in graph
    assert (URIRef(HRN + "addresses"), RDFS.subPropertyOf, URIRef(HRN + "usesDataset")) in graph
