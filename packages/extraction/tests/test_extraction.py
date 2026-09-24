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
    assert HRN.Task not in seed.classes
    assert (HRN.Task, None, None) not in seed.graph
    assert HRN.Task in engine.schema.classes
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
