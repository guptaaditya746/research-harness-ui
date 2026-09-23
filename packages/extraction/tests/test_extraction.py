import json

from harness_extraction import ExtractionEngine
from harness_semantic import hand_written_subschema


class RecordedModel:
    def __init__(self, replies: list[dict]) -> None:
        self.replies = iter(replies)

    def complete_json(self, system: str, user: str) -> dict:
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
