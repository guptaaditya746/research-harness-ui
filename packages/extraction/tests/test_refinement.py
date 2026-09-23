import pytest
from harness_extraction import ExtractedTriple, ExtractionEngine, SchemaInference
from harness_semantic import hand_written_subschema


class Reviewer:
    def __init__(self) -> None:
        self.calls = 0

    def complete_json(self, system: str, user: str) -> dict:
        self.calls += 1
        return {"keep_ids": [0, 999, "1"]}


def test_refinement_cannot_invent_a_triple_or_select_invalid_ids() -> None:
    reviewer = Reviewer()
    engine = ExtractionEngine(reviewer, hand_written_subschema())
    candidates = [
        ExtractedTriple(subject="p4", predicate="title", object="Video Prediction Policy", source_id="p4-metadata", evidence="Video Prediction Policy"),
        ExtractedTriple(subject="p2", predicate="title", object="Other Paper", source_id="p2-metadata", evidence="Other Paper"),
    ]
    selected, report = engine.refine_triples("Which paper introduced Video Prediction Policy?", candidates, [])

    assert selected == candidates[:1]
    assert report == {"candidates": 2, "unique": 2, "kept": 1, "model_calls": 1}
    assert reviewer.calls == 1


class DraftAndVerifier:
    def __init__(self) -> None:
        self.calls = 0

    def complete_json(self, system: str, user: str) -> dict:
        self.calls += 1
        if self.calls == 1:
            return {
                "sparql": "SELECT * WHERE { ?s ?p ?o }",
                "answer": "A different paper introduced the method.",
                "cited_source_ids": ["p4-metadata"],
            }
        return {"supported": False, "answer": "", "cited_source_ids": []}


def test_answer_verifier_rejects_claims_not_supported_by_triples() -> None:
    model = DraftAndVerifier()
    engine = ExtractionEngine(model, hand_written_subschema())
    inferred = SchemaInference(label="Paper", gloss="Paper", classes=["Paper"], properties=["title"])
    triple = ExtractedTriple(subject="p4", predicate="title", object="Video Prediction Policy", source_id="p4-metadata", evidence="Video Prediction Policy")

    with pytest.raises(ValueError, match="no fully supported"):
        engine.answer("Which paper introduced Video Prediction Policy?", inferred, [triple], verify=True)
    assert model.calls == 2


class CanonicalizationModel:
    def __init__(self) -> None:
        self.replies = iter([
            {"triples": [{"subject": "VPP", "predicate": "improves", "object": "41.5% on CALVIN ABC→D", "source_id": "p4-result", "evidence": "41.5% improvement in the Calvin ABC→D benchmark"}]},
            {"definitions": []},
            {"mappings": [{"id": 0, "property": "reportsMetric", "confidence": 0.91}]},
        ])

    def complete_json(self, system: str, user: str) -> dict:
        return next(self.replies)


def test_ontology_review_recovers_a_supported_metric_relation() -> None:
    engine = ExtractionEngine(CanonicalizationModel(), hand_written_subschema())
    inferred = SchemaInference(label="Metric", gloss="Reported result", classes=["Metric"], properties=["reportsMetric"])
    passage = {"passage_id": "p4-result", "paper_id": "p4", "text": "VPP achieves a 41.5% improvement in the Calvin ABC→D benchmark."}

    result, stages = engine.extract_edc("What improvement did VPP report?", inferred, [passage])

    assert [(triple.subject, triple.predicate, triple.object, triple.source_id) for triple in result.triples] == [
        ("p4", "reportsMetric", "41.5% on CALVIN ABC→D", "p4-result")
    ]
    assert stages[-2] == {"stage": "ontology_review", "unresolved": 1, "mapped": 1}
