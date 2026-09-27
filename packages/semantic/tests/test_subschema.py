from harness_semantic import HRN, hand_written_subschema, owlrl_closure, validate_subschema
from rdflib import RDF, RDFS, Graph, URIRef


def test_hand_written_subschema_conforms() -> None:
    schema = hand_written_subschema()
    report = validate_subschema(schema.graph)
    assert report.conforms, report.text
    assert HRN.Paper in schema.classes
    assert HRN.cites in schema.properties
    assert len(owlrl_closure(schema.graph)) >= len(schema.graph)


def test_property_without_range_is_rejected() -> None:
    graph = Graph()
    prop = URIRef("https://harness.local/schema#broken")
    graph.add((prop, RDF.type, RDF.Property))
    graph.add((prop, RDFS.domain, HRN.Paper))
    report = validate_subschema(graph)
    assert not report.conforms
    assert "range" in report.text


def test_seed_is_declared_in_owl2_with_generic_parents_and_origins() -> None:
    from harness_semantic import (
        DBLP,
        HRN,
        TOP_ATTRIBUTE,
        TOP_CLASS,
        TOP_RELATION,
        hand_written_subschema,
        property_kind,
        term_origin,
        validate_subschema,
    )
    from rdflib import OWL, RDF, RDFS

    graph = hand_written_subschema().graph
    assert (HRN.usesDataset, RDF.type, OWL.ObjectProperty) in graph
    assert (HRN.hasTable, RDF.type, OWL.DatatypeProperty) in graph
    assert (HRN.usesDataset, RDFS.subPropertyOf, TOP_RELATION) in graph
    assert (TOP_CLASS, RDF.type, OWL.Class) in graph and (HRN.Paper, RDF.type, OWL.Class) in graph
    assert property_kind(graph, TOP_ATTRIBUTE) == "datatype"
    # DBLP properties get OWL 2 declarations in the merged graph (the seed only says rdf:Property).
    assert property_kind(graph, DBLP.authoredBy) == "object" and property_kind(graph, DBLP.title) == "datatype"
    assert (DBLP.publishedInStream, RDF.type, OWL.ObjectProperty) in graph
    assert [term_origin(i) for i in (DBLP.authoredBy, HRN.title, HRN.usesDataset, HRN.evaluatesAttack)] == [
        "dblp", "dblp", "core", "invented"]
    assert validate_subschema(graph).conforms


def test_no_property_is_both_object_and_datatype_in_the_merged_graph() -> None:
    from harness_semantic import hand_written_subschema, property_kind
    from rdflib import OWL, RDF

    graph = hand_written_subschema().graph
    both = set(graph.subjects(RDF.type, OWL.ObjectProperty)) & set(graph.subjects(RDF.type, OWL.DatatypeProperty))
    assert both == set()
    # Linked properties have the same kind (OWL 2 DL): equivalents and sub-properties.
    from rdflib import RDFS
    for link in (OWL.equivalentProperty, RDFS.subPropertyOf):
        for left, right in graph.subject_objects(link):
            kinds = {property_kind(graph, left), property_kind(graph, right)} - {None}
            assert len(kinds) <= 1, (left, link, right)


def test_the_cskg_content_core_is_named_by_harness_terms_equivalent_to_cskg() -> None:
    from harness_semantic import CSKG, CSKG_INVERSES, TOP_RELATION, term_origin
    from rdflib import OWL

    schema = hand_written_subschema()
    graph = schema.graph
    assert schema.content_core == "cskg-2.0-subset" and len(schema.content_core_sha256 or "") == 64
    assert {HRN.Method, HRN.Task, HRN.Material, HRN.Metric, HRN.OtherEntity} <= schema.classes
    assert (HRN.Method, OWL.equivalentClass, CSKG.Method) in graph
    assert (HRN.Dataset, RDFS.subClassOf, HRN.Material) in graph              # datasets are CS-KG Materials
    assert (CSKG.usedBy, OWL.inverseOf, CSKG.uses) in graph                  # from the subset file
    assert (HRN.uses, OWL.equivalentProperty, CSKG.uses) in graph
    assert (HRN.uses, RDFS.subPropertyOf, TOP_RELATION) in graph
    assert graph.value(HRN.solves, RDFS.range) == HRN.Task
    # The fixed paper-level properties: one name and one type in every run.
    assert (graph.value(HRN.addresses, RDFS.domain), graph.value(HRN.addresses, RDFS.range)) == (HRN.Paper, HRN.Task)
    assert (HRN.evaluatesOn, RDFS.subPropertyOf, HRN.usesDataset) in graph
    assert graph.value(HRN.evaluatesOn, RDFS.range) == HRN.Dataset
    assert {term_origin(HRN.uses), term_origin(HRN.addresses), term_origin(CSKG.uses)} == {"core"}
    assert CSKG_INVERSES["usedBy"] == "uses" and CSKG_INVERSES["outperformedBy"] == "outperforms"
    # CS-KG's base relations have no domain or range; the harness terms carry them, so the seed conforms.
    report = validate_subschema(graph)
    assert report.conforms, report.text
    owlrl_closure(graph)


def test_the_content_core_can_be_switched_off(monkeypatch) -> None:
    from harness_semantic import term_origin

    monkeypatch.setenv("HARNESS_CONTENT_CORE", "none")
    schema = hand_written_subschema()
    assert schema.content_core is None and HRN.Task not in schema.classes and HRN.addresses not in schema.properties
    assert not any(str(s).startswith("https://w3id.org/cskg/") for s in schema.graph.subjects())
    assert term_origin(HRN.addresses) == "invented"


def test_a_missing_cskg_subset_is_an_error(monkeypatch, tmp_path) -> None:
    import pytest

    monkeypatch.setenv("CSKG_SUBSET_PATH", str(tmp_path / "nowhere.ttl"))
    with pytest.raises(FileNotFoundError, match="HARNESS_CONTENT_CORE=none"):
        hand_written_subschema()
