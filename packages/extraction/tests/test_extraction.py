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
