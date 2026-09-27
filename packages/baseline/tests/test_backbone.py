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
    schema_check = next(check for check in result["run"].checks if check.name == "schema")
    assert schema_check.status == "pass" and "triples.ttl parses" in schema_check.detail
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



def test_an_answer_the_verifier_empties_is_answer_unsupported_not_schema_rejected() -> None:
    from harness_extraction import ExtractedTriple, SchemaInference, UnsupportedAnswer

    class Composer:
        def complete_json(self, system: str, user: str) -> dict:
            return {"sparql": "SELECT * WHERE { ?s ?p ?o }", "answer": "Both address robot learning [p4-abstract].",
                    "cited_source_ids": ["p4-abstract"]}

    class Verifier:
        def complete_json(self, system: str, user: str) -> dict:
            return {"removed_claims": ["Both address robot learning."], "supported": False, "answer": "",
                    "cited_source_ids": []}

    engine = ExtractionEngine(Composer(), hand_written_subschema())
    triples = [ExtractedTriple(subject="p4", predicate="addresses", object="robot learning",
                               source_id="p4-abstract", evidence="for robot learning")]
    inferred = SchemaInference(label="Tasks", gloss="Which task", classes=["Paper"], properties=["addresses"])
    with pytest.raises(UnsupportedAnswer, match=r"verifier removed 1 claims.*1 triples available"):
        engine.answer(QUESTION, inferred, triples, verify=True, verifier=Verifier())


def test_an_answer_emptied_with_notes_is_written_again_from_the_triples_alone() -> None:
    from harness_extraction import ExtractedTriple, SchemaInference

    class Composer:
        def __init__(self) -> None:
            self.payloads: list[dict] = []

        def complete_json(self, system: str, user: str) -> dict:
            self.payloads.append(json.loads(user))
            return {"answer": "p4 addresses robot learning [p4-abstract].", "cited_source_ids": ["p4-abstract"]}

    class Verifier:
        def __init__(self) -> None:
            self.calls = 0

        def complete_json(self, system: str, user: str) -> dict:
            self.calls += 1
            if self.calls == 1:
                return {"removed_claims": ["It beats all baselines."], "supported": False, "answer": "",
                        "cited_source_ids": []}
            return {"removed_claims": [], "supported": True, "answer": "p4 addresses robot learning [p4-abstract].",
                    "cited_source_ids": ["p4-abstract"]}

    composer = Composer()
    engine = ExtractionEngine(composer, hand_written_subschema())
    triples = [ExtractedTriple(subject="p4", predicate="usesDataset", object="robot learning",
                               source_id="p4-abstract", evidence="for robot learning")]
    inferred = SchemaInference(label="Tasks", gloss="Which task", classes=["Paper"], properties=["usesDataset"])
    report: dict = {}
    result = engine.answer(QUESTION, inferred, triples, verify=True, verifier=Verifier(),
                           notes=["It beats all baselines."], report=report)
    assert report == {"repaired": True, "notes_dropped": True}
    assert "research_notes" in composer.payloads[0] and "research_notes" not in composer.payloads[1]
    assert result.cited_source_ids == ["p4-abstract"] and "hrn:usesDataset" in result.sparql


def test_an_answer_citing_no_triple_source_is_repaired_once_without_notes() -> None:
    from harness_extraction import ExtractedTriple, SchemaInference, UnsupportedAnswer

    systems: list[str] = []

    class Composer:
        calls = 0

        def complete_json(self, system: str, user: str) -> dict:
            Composer.calls += 1
            systems.append(system)
            return {"answer": "p9 does it [p9-abstract]. Notes say so [research_notes].",
                    "cited_source_ids": ["p9-abstract"]}

    engine = ExtractionEngine(Composer(), hand_written_subschema())
    triples = [ExtractedTriple(subject="p4", predicate="usesDataset", object="x", source_id="p4-abstract", evidence="x")]
    inferred = SchemaInference(label="l", gloss="g", classes=["Paper"], properties=["usesDataset"])
    report: dict = {}
    with pytest.raises(UnsupportedAnswer) as caught:
        engine.answer(QUESTION, inferred, triples, notes=["a note"], report=report)
    assert caught.value.stage == "citation" and Composer.calls == 2
    assert caught.value.unknown_sources == ["p9-abstract", "research_notes"]
    # The repair names the ids that are no triple's source.
    assert '"research_notes"' in systems[1] and '"research_notes"' not in systems[0]
    assert report["repaired"] is True and "citations_dropped" not in report


def test_sentences_citing_only_unknown_ids_are_dropped_and_the_rest_kept() -> None:
    from harness_extraction import ExtractedTriple, SchemaInference

    class Composer:
        def complete_json(self, system: str, user: str) -> dict:
            return {"answer": "p4 uses x [p4-abstract]. The agent also found y [research_notes]. "
                              "It scores [0.5-0.7] on z [p4-abstract, p9-x].",
                    "cited_source_ids": ["p4-abstract", "research_notes"]}

    engine = ExtractionEngine(Composer(), hand_written_subschema())
    triples = [ExtractedTriple(subject="p4", predicate="usesDataset", object="x", source_id="p4-abstract", evidence="x")]
    inferred = SchemaInference(label="l", gloss="g", classes=["Paper"], properties=["usesDataset"])
    report: dict = {}
    result = engine.answer(QUESTION, inferred, triples, report=report)
    assert result.answer == "p4 uses x [p4-abstract]. It scores [0.5-0.7] on z [p4-abstract]."
    assert result.cited_source_ids == ["p4-abstract"]
    assert report == {"citations_dropped": {"sentences": 1, "ids": ["research_notes", "p9-x"]}}


def test_an_answer_nothing_supports_after_the_repair_states_the_extracted_facts(tmp_path: Path) -> None:
    from harness_extraction import ExtractedTriple, SchemaInference

    class Rejecting:
        calls = 0

        def complete_json(self, system: str, user: str) -> dict:
            Rejecting.calls += 1
            if Rejecting.calls % 2:
                return {"answer": "Everything is solved [p4-abstract].", "cited_source_ids": ["p4-abstract"]}
            return {"sentences": [{"text": "Everything is solved [p4-abstract].", "supported": False,
                                   "source_ids": [], "reason": "overclaims"}],
                    "removed_claims": ["Everything is solved."], "supported": False, "answer": "", "cited_source_ids": []}

    engine = ExtractionEngine(Rejecting(), hand_written_subschema())
    runner = Baseline20(two_papers(tmp_path), engine, enhanced=True, refine=True, refinement_model=Rejecting())
    runner.started = __import__("time").monotonic()
    state = {"question": QUESTION, "passages": [], "concept": None,
             "inferred": SchemaInference(label="Tasks", gloss="g", classes=["Paper"], properties=["usesDataset"]),
             "extracted": [ExtractedTriple(subject="p4", predicate="usesDataset", object="robot videos",
                                           source_id="p4-abstract", evidence="robot videos")]}
    answer = runner._query_answer(state)["answer"]
    assert answer.answer == "Video Prediction Policy: uses dataset robot videos [p4-abstract]."
    assert [item["verdict"] for item in answer.sentences] == ["removed", "from_facts"]
    warn = [check for check in runner.checks if check.name == "answer composition"]
    assert warn and warn[0].status == "warn" and "states 1 of the 1 extracted facts" in warn[0].detail


def test_an_answer_that_cites_only_research_notes_after_the_repair_states_the_extracted_facts(tmp_path: Path) -> None:
    from harness_extraction import ExtractedTriple, SchemaInference

    class NotesOnly:
        def complete_json(self, system: str, user: str) -> dict:
            return {"answer": "The agent found it works [research_notes].", "cited_source_ids": ["research_notes"]}

    engine = ExtractionEngine(NotesOnly(), hand_written_subschema())
    runner = Baseline20(two_papers(tmp_path), engine, enhanced=True, refine=True, refinement_model=NotesOnly())
    runner.started = __import__("time").monotonic()
    state = {"question": QUESTION, "passages": [], "concept": None,
             "inferred": SchemaInference(label="Tasks", gloss="g", classes=["Paper"], properties=["usesDataset"]),
             "extracted": [ExtractedTriple(subject="p4", predicate="usesDataset", object="robot videos",
                                           source_id="p4-abstract", evidence="robot videos")]}
    answer = runner._query_answer(state)["answer"]
    assert answer.answer == "Video Prediction Policy: uses dataset robot videos [p4-abstract]."
    details = [check.detail for check in runner.checks if check.name == "answer composition"]
    assert any("cited research_notes" in detail and "states 1 of the 1" in detail for detail in details)


def test_citation_edge_cases_keep_prose_brackets_and_attach_trailing_citations() -> None:
    from harness_extraction.engine import drop_unknown_citations

    sources = {"p4-abstract", "p2-s1-p3"}
    # A citation after the full stop belongs to the sentence before it; a notes-only sentence goes.
    assert drop_unknown_citations("A is true. [research_notes] B is true. [p4-abstract]", [], sources)[:2] == (
        "B is true [p4-abstract].", ["p4-abstract"])
    # A claim cited only by an unknown id is not re-attributed to the known id later in the sentence.
    assert drop_unknown_citations("Claim A [p9-abstract] and claim B [p4-abstract].", [], sources)[0] == ""
    # Abbreviations do not end sentences; model names, ranges and tokens in brackets are prose.
    text = "Smith et al. report it [p4-abstract]. It beats GPT-4 [GPT-4], e.g. on [0.5-0.7] [p2-s1-p3]."
    assert drop_unknown_citations(text, [], sources) == (text, ["p4-abstract", "p2-s1-p3"], [], 0)
    assert drop_unknown_citations("Uses [p4-abstract, Table 2].", [], sources)[0] == "Uses [p4-abstract]."


def test_an_uncited_answer_is_not_repaired_with_the_same_prompt() -> None:
    from harness_extraction import ExtractedTriple, SchemaInference

    class Honest:
        calls = 0

        def complete_json(self, system: str, user: str) -> dict:
            Honest.calls += 1
            return {"answer": "The triples do not say which method is fastest.", "cited_source_ids": []}

    engine = ExtractionEngine(Honest(), hand_written_subschema())
    triples = [ExtractedTriple(subject="p4", predicate="usesDataset", object="x", source_id="p4-abstract", evidence="x")]
    inferred = SchemaInference(label="l", gloss="g", classes=["Paper"], properties=["usesDataset"])
    result = engine.answer(QUESTION, inferred, triples, report={})
    assert result.answer.startswith("The triples do not say") and Honest.calls == 1


def test_a_fast_run_without_evidence_review_does_not_fall_back_to_stating_facts(tmp_path: Path) -> None:
    from harness_extraction import ExtractedTriple, SchemaInference, UnsupportedAnswer

    class NotesOnly:
        def complete_json(self, system: str, user: str) -> dict:
            return {"answer": "It works [research_notes].", "cited_source_ids": ["research_notes"]}

    runner = Baseline20(two_papers(tmp_path), ExtractionEngine(NotesOnly(), hand_written_subschema()))
    runner.started = __import__("time").monotonic()
    state = {"question": QUESTION, "passages": [], "concept": None,
             "inferred": SchemaInference(label="Tasks", gloss="g", classes=["Paper"], properties=["usesDataset"]),
             "extracted": [ExtractedTriple(subject="p4", predicate="usesDataset", object="robot videos",
                                           source_id="p4-abstract", evidence="robot videos")]}
    with pytest.raises(UnsupportedAnswer):
        runner._query_answer(state)
