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
