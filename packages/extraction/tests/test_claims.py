from harness_extraction.claims import build_claims, classify_claim_relations, derive_query_claim, enrich_result_claims, valid_edge
from harness_extraction import ExtractionEngine
from harness_schema.models import Triple
from harness_semantic import hand_written_subschema
from rdflib import URIRef


def triple(object_: str, source: str = "p4-results") -> Triple:
    return Triple(s="p4", p="reportsMetric", o=object_, source=source,
                  evidence=object_, support="one passage")


def test_result_claims_keep_different_values_and_provenance() -> None:
    claims, relations = build_claims([
        triple("41.5% improvement in success rate", "p4-results"),
        triple("18.6% improvement in success rate", "p4-abstract"),
    ], lambda _source, _subject: "https://harness.local/resource/p4")
    assert len(claims) == 2
    assert {claim.fields["value"].value for claim in claims} == {"41.5", "18.6"}
    assert {claim.source_ids[0] for claim in claims} == {"p4-results", "p4-abstract"}
    assert all(claim.evidence_anchors[0].kind == "passage" for claim in claims)
    assert all(claim.fields["value"].evidence_ids for claim in claims)
    assert len(relations) == 1
    assert relations[0].relation == "possibleConflict"


def test_duplicate_assertion_merges_sources_without_losing_field_provenance() -> None:
    claims, _ = build_claims([
        triple("41.5% improvement in success rate", "p4-results"),
        triple("41.5% improvement in success rate", "p4-abstract"),
    ], lambda _source, _subject: "https://harness.local/resource/p4")
    assert len(claims) == 1
    assert claims[0].fields["value"].source_ids == ["p4-abstract", "p4-results"]


def test_placeholder_edge_is_rejected() -> None:
    assert not valid_edge(triple("null"))


def test_context_enrichment_requires_verbatim_source_quote() -> None:
    claims, _ = build_claims([triple("41.5% improvement", "p4-results")],
                             lambda _source, _subject: "paper-p4")
    class Model:
        def complete_json(self, _system: str, _user: str) -> dict:
            return {"enrichments": [{"claim_id": claims[0].id, "fields": {
                "benchmark": {"value": "CALVIN", "source_id": "p4-results",
                              "quote": "on CALVIN ABC-D"},
                "comparator": {"value": "Baseline Y", "source_id": "p4-results",
                               "quote": "compared with Baseline Y"},
            }}]}
    enriched = enrich_result_claims(claims, [{"passage_id": "p4-results",
                                             "text": "It improved 41.5% on CALVIN ABC-D."}], Model())
    assert enriched[0].fields["benchmark"].value == "CALVIN"
    assert "comparator" not in enriched[0].fields


def test_dynamic_relation_is_added_to_the_run_subschema() -> None:
    engine = ExtractionEngine(object(), hand_written_subschema())
    engine.register_local_relation("evaluatedInEnvironment", "Evaluation setting environment")
    iri = URIRef("https://harness.local/schema#evaluatedInEnvironment")
    assert iri in engine.schema.properties
    assert "evaluatedInEnvironment" in engine.property_names


def test_matching_material_context_confirms_a_numeric_conflict() -> None:
    claims, relations = build_claims([
        triple("41.5% success rate on CALVIN in Standard setting", "p4-results"),
        triple("18.6% success rate on CALVIN in Standard setting", "p4-abstract"),
    ], lambda _source, _subject: "paper-p4")
    for claim in claims:
        fields = dict(claim.fields)
        for name, value in {"benchmark": "CALVIN", "setting": "Standard"}.items():
            fields[name] = fields["value"].model_copy(update={"value": value})
        claim_index = claims.index(claim)
        claims[claim_index] = claim.model_copy(update={"fields": fields})
    assert classify_claim_relations(claims)[0].relation == "confirmedConflict"


def test_cross_paper_question_creates_a_separate_synthesis_claim() -> None:
    claims, _ = build_claims([triple("uses object-centric representation")], lambda _source, _subject: "paper-p4")
    synthesis = derive_query_claim("What is shared across these papers?", claims)
    assert synthesis is not None
    assert synthesis.kind == "SynthesisClaim"
    assert synthesis.supported_by == [claims[0].id]
