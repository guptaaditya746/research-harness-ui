import pytest
from harness_graphstore import (
    InvariantViolation,
    ProvenanceDataset,
    analyse_query,
    check_answer_claims,
    sparql_check,
)
from harness_ledger import Ledger, Obligation
from harness_semantic import HRN, hand_written_subschema
from hypothesis import given
from hypothesis import strategies as st
from rdflib import Literal, URIRef


def structural(index: int, prop: str = str(HRN.cites)) -> Obligation:
    return Obligation(
        id=f"o-{index}",
        subject=f"https://harness.local/resource/p{index}",
        property=prop,
        kind="structural",
    )


def test_named_graph_provenance_support_counts_and_exports() -> None:
    schema = hand_written_subschema()
    store = ProvenanceDataset(schema, {"p1-s1-p1", "p1-s2-p1"})
    triple = (URIRef("https://harness.local/resource/p1"), HRN.title, Literal("A paper"))
    store.add_event("e1", [triple], ["p1-s1-p1"])
    store.add_event("e2", [triple], ["p1-s2-p1"])
    assert store.support_count(triple) == 2
    assert store.provenance(triple) == {"p1-s1-p1", "p1-s2-p1"}
    assert "wasDerivedFrom" in store.export_trig()
    assert "A paper" in store.export_turtle()


def test_build_validation_rejects_object_for_literal_property() -> None:
    store = ProvenanceDataset(hand_written_subschema(), {"p1-metadata"})
    with pytest.raises(ValueError, match="literal"):
        store.add_event(
            "e1",
            [
                (
                    URIRef("https://harness.local/resource/p1"),
                    HRN.title,
                    URIRef("https://example.test/not-a-literal"),
                )
            ],
            ["p1-metadata"],
        )
    assert len(store.dataset) == 0


def test_invariant_2_rejects_missing_provenance_source() -> None:
    store = ProvenanceDataset(hand_written_subschema(), {"known-passage"})
    with pytest.raises(InvariantViolation, match="invariant:2"):
        store.add_event(
            "e1",
            [(URIRef("https://harness.local/resource/p1"), HRN.title, Literal("A"))],
            ["missing-passage"],
        )


def test_invariant_3_rejects_unknown_predicate_and_class() -> None:
    schema = hand_written_subschema()
    with pytest.raises(InvariantViolation, match="invariant:3"):
        sparql_check("SELECT ?s WHERE { ?s <https://example.test/unknown> ?o }", schema)
    with pytest.raises(InvariantViolation, match="invariant:3"):
        sparql_check("SELECT ?s WHERE { ?s a <https://example.test/UnknownClass> }", schema)


def test_query_analysis_builds_structural_ledger_from_domain_facts() -> None:
    query = """PREFIX hrn: <https://harness.local/schema#>
    SELECT ?paper WHERE { ?paper hrn:cites <https://harness.local/resource/p17> }"""
    facts = [
        {"subject": "p1", "predicate": "title", "object": "One", "source_id": "p1-meta"},
        {"subject": "p2", "predicate": "title", "object": "Two", "source_id": "p2-meta"},
        {"subject": "p1", "predicate": "cites", "object": "p17", "source_id": "p1-ref-2"},
    ]
    plan = analyse_query(query, hand_written_subschema(), facts)
    assert plan.shape == "set"
    assert len(plan.obligations) == 2
    assert plan.obligations[0].expected_object == "https://harness.local/resource/p17"
    ledger = Ledger(plan.obligations)
    ledger.close_structural(facts)
    assert ledger.closed
    assert ledger.obligations[0].evidence == ["p1-ref-2"]


def test_invariant_4_requires_rows_and_resolvable_passages() -> None:
    with pytest.raises(InvariantViolation, match="invariant:4"):
        check_answer_claims(
            [{"text": "Unsupported", "row_ids": [], "triple_ids": ["t1"]}],
            {"row-1"},
            {"t1": {"p1-s1-p1"}},
            {"p1-s1-p1"},
        )
    with pytest.raises(InvariantViolation, match="invariant:4"):
        check_answer_claims(
            [{"text": "Bad citation", "row_ids": ["row-1"], "triple_ids": ["t1"]}],
            {"row-1"},
            {"t1": {"missing"}},
            {"p1-s1-p1"},
        )


def test_invariant_5_blocks_query_until_ledger_is_terminal() -> None:
    schema = hand_written_subschema()
    store = ProvenanceDataset(schema, {"p1-metadata"})
    store.add_event(
        "e1",
        [(URIRef("https://harness.local/resource/p1"), HRN.title, Literal("One"))],
        ["p1-metadata"],
    )
    ledger = Ledger([structural(1, str(HRN.title))])
    query = (
        "PREFIX hrn: <https://harness.local/schema#> SELECT ?title WHERE { ?s hrn:title ?title }"
    )
    with pytest.raises(InvariantViolation, match="invariant:5"):
        store.query(query, ledger, "evaluated 0 of 1 papers")
    ledger.satisfy("o-1", ["p1-metadata"])
    assert next(iter(store.query(query, ledger, "evaluated 1 of 1 papers")))[0] == Literal("One")


@pytest.mark.parametrize(
    ("query", "shape"),
    [
        (
            "SELECT (COUNT(?s) AS ?n) WHERE { ?s <https://harness.local/schema#title> ?o }",
            "aggregate",
        ),
        (
            "SELECT ?o WHERE { <https://harness.local/resource/p1> <https://harness.local/schema#title> ?o }",
            "point",
        ),
        ("SELECT ?o WHERE { ?s <https://harness.local/schema#cites>+ ?o }", "path"),
    ],
)
def test_query_shape_classification(query: str, shape: str) -> None:
    assert analyse_query(query, hand_written_subschema(), []).shape == shape


@given(st.lists(st.integers(min_value=1, max_value=500), min_size=1, max_size=20, unique=True))
def test_query_analysis_enumerates_every_domain_member(paper_numbers: list[int]) -> None:
    facts = [
        {
            "subject": f"p{number}",
            "predicate": "title",
            "object": f"Paper {number}",
            "source_id": f"p{number}-metadata",
        }
        for number in paper_numbers
    ]
    query = "SELECT ?paper WHERE { ?paper <https://harness.local/schema#cites> ?other }"
    plan = analyse_query(query, hand_written_subschema(), facts)
    assert len(plan.obligations) == len(paper_numbers)
    assert {item.subject.rsplit("/", 1)[-1] for item in plan.obligations} == {
        f"p{number}" for number in paper_numbers
    }


def test_class_enumeration_creates_type_obligations_closed_from_metadata() -> None:
    facts = [{"subject": "p1", "predicate": "title", "object": "One", "source_id": "p1-meta"}]
    query = "SELECT ?paper WHERE { ?paper a <https://harness.local/schema#Paper> }"
    plan = analyse_query(query, hand_written_subschema(), facts)
    assert plan.properties == (str(URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")),)
    ledger = Ledger(plan.obligations)
    ledger.close_structural(facts)
    assert ledger.closed
