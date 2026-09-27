import json

from harness_extraction import ExtractionEngine
from harness_semantic import hand_written_subschema


class RecordedModel:
    def __init__(self, replies: list[dict]) -> None:
        self.replies = iter(replies)
        self.calls = 0

    def complete_json(self, system: str, user: str) -> dict:
        self.calls += 1
        json.loads(user)
        return next(self.replies)


def test_schema_inference_extends_seed_with_run_local_terms() -> None:
    from harness_semantic import HRN, validate_subschema
    from rdflib import RDFS

    seed = hand_written_subschema()
    schema_model = RecordedModel([{
        "label": "Tasks and approaches", "gloss": "Research themes",
        "classes": ["Paper", "Task", "Approach"],
        "properties": ["studiesTask", "usesApproach"],
        "class_definitions": [
            {"name": "Task", "definition": "An NLP problem studied by a paper."},
            {"name": "Approach", "definition": "A method used to address a task."},
        ],
        "property_definitions": [
            {"name": "studiesTask", "definition": "A paper studies a task.",
             "domain": "Paper", "range": "Task"},
            {"name": "usesApproach", "definition": "A paper uses an approach.",
             "domain": "Paper", "range": "Approach"},
        ],
    }])
    engine = ExtractionEngine(
        RecordedModel([]), seed, schema_model=schema_model,
    )

    inferred = engine.infer_schema(
        "Which NLP tasks appear most often, and what kinds of approaches are used for them?",
        [{"passage_id": "p4-intro", "paper_id": "p4", "text": "Tasks and approaches."}],
    )

    assert inferred.classes == ["Paper", "Task", "Approach"]
    assert inferred.properties == ["studiesTask", "usesApproach"]
    assert schema_model.calls == 1
    assert HRN.Task in seed.classes                     # a content-core (CS-KG) class since Phase 4
    assert HRN.Approach not in seed.classes
    assert (HRN.Approach, None, None) not in seed.graph
    assert HRN.Approach in engine.schema.classes
    assert engine.schema.graph.value(HRN.studiesTask, RDFS.range) == HRN.Task
    assert engine.schema.graph.value(HRN.usesApproach, RDFS.range) == HRN.Approach
    assert validate_subschema(engine.schema.graph).conforms


def test_schema_inference_accepts_valid_new_terms_without_definitions() -> None:
    from harness_semantic import HRN
    from rdflib import RDFS

    schema_model = RecordedModel([{
        "label": "Tasks and approaches", "gloss": "Research themes",
        "classes": ["Paper", "Task", "Approach"],
        "properties": ["studiesTask", "usesApproach"],
    }])
    engine = ExtractionEngine(
        RecordedModel([]), hand_written_subschema(), schema_model=schema_model,
    )

    inferred = engine.infer_schema(
        "Which NLP tasks appear most often?",
        [{"passage_id": "p4-intro", "paper_id": "p4", "text": "The paper studies NER."}],
    )

    assert inferred.classes == ["Paper", "Task", "Approach"]
    assert inferred.properties == ["studiesTask", "usesApproach"]
    assert engine.schema.graph.value(HRN.studiesTask, RDFS.range) == HRN.Task
    assert engine.schema.graph.value(HRN.usesApproach, RDFS.range) == HRN.Approach
    assert schema_model.calls == 1


def test_schema_inference_accepts_many_query_local_classes() -> None:
    names = [f"TaskType{index}" for index in range(10)]
    schema_model = RecordedModel([{
        "label": "NLP task types", "gloss": "Tasks compared across papers",
        "classes": ["Paper", *names], "properties": ["claims"],
    }])
    engine = ExtractionEngine(
        RecordedModel([]), hand_written_subschema(), schema_model=schema_model,
    )

    inferred = engine.infer_schema(
        "Which NLP tasks recur?",
        [{"passage_id": "p4-intro", "paper_id": "p4", "text": "The paper studies tasks."}],
    )

    assert set(names) <= set(inferred.classes)
    assert all(name in engine.class_names for name in names)


def test_extraction_discards_sources_outside_inputs() -> None:
    passage = {"passage_id": "p4-abstract", "paper_id": "p4", "text": "VPP is proposed."}
    engine = ExtractionEngine(
        RecordedModel(
            [
                {
                    "label": "Introduced method",
                    "gloss": "Paper proposing a method",
                    "classes": ["Paper"],
                    "properties": ["claims"],
                },
                {
                    "triples": [
                        {
                            "subject": "p4",
                            "predicate": "claims",
                            "object": "VPP",
                            "source_id": "missing",
                            "evidence": "VPP is proposed.",
                        }
                    ]
                },
            ]
        ),
        hand_written_subschema(),
    )
    inferred = engine.infer_schema("Which paper introduced VPP?", [passage])
    assert engine.extract("Which paper introduced VPP?", inferred, [passage]).triples == []


def test_schema_inference_can_use_a_dedicated_model() -> None:
    schema_model = RecordedModel([{
        "label": "Introduced method",
        "gloss": "Paper proposing a method",
        "classes": ["Paper"],
        "properties": ["claims"],
    }])
    claim_model = RecordedModel([{"triples": []}])
    engine = ExtractionEngine(
        claim_model, hand_written_subschema(), schema_model=schema_model,
    )
    passage = {"passage_id": "p4-abstract", "paper_id": "p4", "text": "VPP is proposed."}

    inferred = engine.infer_schema("Which paper introduced VPP?", [passage])
    engine.extract("Which paper introduced VPP?", inferred, [passage])

    assert schema_model.calls == 1
    assert claim_model.calls == 1


class PayloadModel(RecordedModel):
    def complete_json(self, system: str, user: str) -> dict:
        self.payloads = [*getattr(self, "payloads", []), json.loads(user)]
        return super().complete_json(system, user)


def test_agent_notes_are_their_own_field_and_stay_out_of_definitions_and_verification() -> None:
    from harness_extraction import ExtractedTriple
    from harness_semantic import HRN
    from rdflib import RDFS

    schema_model = PayloadModel([{"label": "Tasks", "gloss": "Tasks", "classes": ["Paper", "Task"],
                                  "properties": ["studiesTask"]}])
    answer_model = PayloadModel([
        {"sparql": "SELECT ?p WHERE { ?p ?q ?o }", "answer": "p4 studies NER [p4-intro].",
         "cited_source_ids": ["p4-intro"]},
        {"supported": True, "answer": "p4 studies NER [p4-intro].", "cited_source_ids": ["p4-intro"]},
    ])
    engine = ExtractionEngine(answer_model, hand_written_subschema(), schema_model=schema_model)
    notes = ["p4 studies named entity recognition."]
    inferred = engine.infer_schema("Which tasks?", [{"passage_id": "p4-intro", "paper_id": "p4", "text": "NER."}],
                                   notes=notes)
    assert schema_model.payloads[0]["research_notes"] == notes
    assert list(schema_model.payloads[0])[-2:] == ["research_notes", "question"]
    assert "named entity" not in str(engine.schema.graph.value(HRN.studiesTask, RDFS.comment))
    triple = ExtractedTriple(subject="p4", predicate="studiesTask", object="NER", source_id="p4-intro",
                             evidence="NER.")
    engine.answer("Which tasks?", inferred, [triple], verify=True, notes=notes)
    composition, verification = answer_model.payloads
    assert composition["research_notes"] == notes and "research_notes" not in verification
    assert verification["question"] == "Which tasks?"


def test_an_answer_cut_off_at_the_output_limit_is_retried_once_in_a_shorter_format() -> None:
    import pytest
    from harness_extraction import ExtractedTriple, ModelFailure
    from harness_extraction.schemas import SchemaInference

    class CutOffOnce:
        def __init__(self, failures: int) -> None:
            self.failures = failures
            self.systems: list[str] = []

        def complete_json(self, system: str, user: str) -> dict:
            self.systems.append(system)
            if len(self.systems) <= self.failures:
                raise ModelFailure("unusable reply: reply hit the 4096-token output limit before finishing")
            return {"sparql": "SELECT ?p WHERE { ?p ?q ?o }", "answer": "NER [p4-intro].",
                    "cited_source_ids": ["p4-intro"]}

    inferred = SchemaInference(label="t", gloss="t", classes=["Paper"], properties=["title"])
    triple = ExtractedTriple(subject="p4", predicate="title", object="NER", source_id="p4-intro", evidence="NER.")
    model = CutOffOnce(1)
    engine = ExtractionEngine(model, hand_written_subschema())
    assert engine.answer("Which tasks?", inferred, [triple]).answer == "NER [p4-intro]."
    assert "Keep it short" not in model.systems[0] and "Keep it short" in model.systems[1]
    stubborn = CutOffOnce(2)
    with pytest.raises(ModelFailure):
        ExtractionEngine(stubborn, hand_written_subschema()).answer("Which tasks?", inferred, [triple])
    assert len(stubborn.systems) == 2


def test_a_cut_off_reply_says_how_it_ended() -> None:
    from harness_extraction.model_client import _cut_off_ending

    assert "JSON-mode loop" in _cut_off_ending('{"answer": "x"' + " " * 500)
    assert _cut_off_ending('{"answer": "a b c d').endswith('{"answer": "a b c d')


def test_v2_prompts_see_property_signatures_and_v1_prompts_do_not(monkeypatch) -> None:
    schema_reply = {"label": "x", "gloss": "y", "classes": ["Paper", "Task"], "properties": ["addresses"]}
    passages = [{"passage_id": "p99-s1", "paper_id": "p99", "text": "We study BenchY."}]
    sent: list[dict] = []

    class Recording(RecordedModel):
        def complete_json(self, system: str, user: str) -> dict:
            sent.append(json.loads(user))
            return super().complete_json(system, user)

    engine = ExtractionEngine(Recording([{"triples": []}]), hand_written_subschema(),
                              schema_model=Recording([schema_reply]))
    inferred = engine.infer_schema("Which problems does p99 study?", passages)
    engine.extract("Which problems does p99 study?", inferred, passages)
    signatures = sent[0]["property_signatures"]
    assert any(row.startswith("addresses: Paper -> Task. ") for row in signatures)
    assert any(row.startswith("solves: ResearchEntity -> Task. ") for row in signatures)
    assert sent[1]["property_signatures"] == [row for row in signatures if row.startswith("addresses:")]

    monkeypatch.setenv("HARNESS_PROMPT_SET", "v1")
    sent.clear()
    engine = ExtractionEngine(Recording([]), hand_written_subschema(), schema_model=Recording([schema_reply]))
    engine.infer_schema("Which problems does p99 study?", passages)
    assert "property_signatures" not in sent[0]
