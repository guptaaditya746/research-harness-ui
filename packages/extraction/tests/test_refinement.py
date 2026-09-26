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
        if self.calls % 2:                      # drafts (first answer, then the one repair)
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

    with pytest.raises(ValueError, match="no fully supported.*after one repair"):
        engine.answer("Which paper introduced Video Prediction Policy?", inferred, [triple], verify=True)
    assert model.calls == 4                     # draft, verdict, repaired draft, verdict


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


class SentenceVerifier:
    def __init__(self) -> None:
        self.calls = 0

    def complete_json(self, system: str, user: str) -> dict:
        self.calls += 1
        if self.calls == 1:
            return {"answer": "MethodX is introduced by p4 [p4-metadata]. It beats every baseline [p4-metadata].",
                    "cited_source_ids": ["p4-metadata"]}
        return {"sentences": [
            {"text": "MethodX is introduced by p4 [p4-metadata].", "supported": True, "source_ids": ["p4-metadata"]},
            {"text": "It beats every baseline [p4-metadata].", "supported": False, "source_ids": [],
             "reason": "no triple compares with baselines"},
            {"text": "It is fast.", "supported": True, "source_ids": ["p9-unknown"]},
        ], "removed_claims": ["It beats every baseline."], "supported": True,
            "answer": "MethodX is introduced by p4 [p4-metadata].", "cited_source_ids": ["p4-metadata"]}


def test_the_answer_is_the_accepted_sentences_and_every_verdict_is_kept() -> None:
    engine = ExtractionEngine(SentenceVerifier(), hand_written_subschema())
    inferred = SchemaInference(label="Paper", gloss="Paper", classes=["Paper"], properties=["title"])
    triple = ExtractedTriple(subject="p4", predicate="title", object="MethodX", source_id="p4-metadata", evidence="MethodX")
    result = engine.answer("Who introduced MethodX?", inferred, [triple], verify=True)
    assert result.answer == "MethodX is introduced by p4 [p4-metadata]."
    assert [item["verdict"] for item in result.sentences] == ["accepted", "removed", "removed"]
    assert result.sentences[2]["reason"] == "cites no source of the supplied triples"   # supported, but no known source


def test_sentence_citations_are_checked_and_added_and_provenance_comes_from_them() -> None:
    class Verifier:
        calls = 0

        def complete_json(self, system: str, user: str) -> dict:
            Verifier.calls += 1
            if Verifier.calls == 1:
                return {"answer": "MethodX is by p4 [p4-metadata].", "cited_source_ids": ["p4-metadata"]}
            return {"sentences": [
                {"text": "MethodX is by p4.", "supported": True, "source_ids": ["[p4-metadata]"]},
                {"text": "It appeared at ACL [p7-venue].", "supported": True, "source_ids": ["p4-metadata"]},
            ], "supported": True, "answer": "MethodX is by p4.", "cited_source_ids": []}

    engine = ExtractionEngine(Verifier(), hand_written_subschema())
    inferred = SchemaInference(label="Paper", gloss="Paper", classes=["Paper"], properties=["title"])
    triple = ExtractedTriple(subject="p4", predicate="title", object="MethodX", source_id="p4-metadata", evidence="MethodX")
    result = engine.answer("Who made MethodX?", inferred, [triple], verify=True)
    assert result.answer == "MethodX is by p4 [p4-metadata]."            # bracket added, id normalised
    assert result.cited_source_ids == ["p4-metadata"]                     # from the accepted sentence
    assert result.sentences[1]["verdict"] == "removed" and "p7-venue" in result.sentences[1]["reason"]
