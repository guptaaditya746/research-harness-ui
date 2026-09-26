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
