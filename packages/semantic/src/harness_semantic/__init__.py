"""M2 hand-written schema and its static SHACL constraints."""

from __future__ import annotations

from dataclasses import dataclass

from owlrl import DeductiveClosure, OWLRL_Semantics
from pyshacl import validate
from rdflib import OWL, RDF, RDFS, SH, XSD, Graph, Literal, Namespace, URIRef

HRN = Namespace("https://harness.local/schema#")


@dataclass(frozen=True)
class Subschema:
    graph: Graph
    classes: frozenset[URIRef]
    properties: frozenset[URIRef]
    structural_properties: frozenset[URIRef]


@dataclass(frozen=True)
class ValidationReport:
    conforms: bool
    graph: Graph
    text: str


def _add_class(graph: Graph, iri: URIRef) -> None:
    graph.add((iri, RDF.type, RDFS.Class))
    graph.add((iri, RDFS.label, Literal(str(iri).rsplit("#", 1)[-1])))


def _add_property(graph: Graph, iri: URIRef, domain: URIRef, range_: URIRef) -> None:
    graph.add((iri, RDF.type, RDF.Property))
    graph.add((iri, RDFS.label, Literal(str(iri).rsplit("#", 1)[-1])))
    graph.add((iri, RDFS.domain, domain))
    graph.add((iri, RDFS.range, range_))


def hand_written_subschema() -> Subschema:
    """Return the small schema used before seed lookup and minting arrive in M4."""
    graph = Graph()
    graph.bind("hrn", HRN)
    graph.bind("xsd", XSD)
    classes = frozenset({HRN.Paper, HRN.Author, HRN.Venue, HRN.Dataset, HRN.Metric, HRN.Claim})
    for iri in classes:
        _add_class(graph, iri)

    specs = {
        HRN.title: (HRN.Paper, XSD.string),
        HRN.authoredBy: (HRN.Paper, HRN.Author),
        HRN.year: (HRN.Paper, XSD.gYear),
        HRN.venue: (HRN.Paper, HRN.Venue),
        HRN.cites: (HRN.Paper, HRN.Paper),
        HRN.sectionOf: (HRN.Claim, HRN.Paper),
        HRN.hasTable: (HRN.Paper, XSD.string),
        HRN.usesDataset: (HRN.Paper, HRN.Dataset),
        HRN.reportsMetric: (HRN.Paper, HRN.Metric),
        HRN.claims: (HRN.Paper, HRN.Claim),
        HRN.comparesTo: (HRN.Paper, HRN.Paper),
    }
    for iri, (domain, range_) in specs.items():
        _add_property(graph, iri, domain, range_)
    structural = frozenset(
        {HRN.title, HRN.authoredBy, HRN.year, HRN.venue, HRN.cites, HRN.sectionOf, HRN.hasTable}
    )
    return Subschema(graph, classes, frozenset(specs), structural)


def schema_shapes() -> Graph:
    shapes = Graph()
    shape = HRN.PropertyShape
    shapes.add((shape, RDF.type, SH.NodeShape))
    shapes.add((shape, SH.targetClass, RDF.Property))
    for index, predicate in enumerate((RDFS.domain, RDFS.range), start=1):
        constraint = URIRef(f"{shape}/constraint/{index}")
        shapes.add((shape, SH.property, constraint))
        shapes.add((constraint, SH.path, predicate))
        shapes.add((constraint, SH.minCount, Literal(1)))
        shapes.add((constraint, SH.maxCount, Literal(1)))
    return shapes


def validate_subschema(graph: Graph) -> ValidationReport:
    conforms, report_graph, report_text = validate(
        data_graph=graph,
        shacl_graph=schema_shapes(),
        inference="none",
        advanced=True,
    )
    return ValidationReport(bool(conforms), report_graph, str(report_text))


def owlrl_closure(graph: Graph) -> Graph:
    expanded = Graph()
    for triple in graph:
        expanded.add(triple)
    DeductiveClosure(OWLRL_Semantics).expand(expanded)
    if any(expanded.triples((None, RDF.type, OWL.Nothing))):
        raise ValueError("OWL RL closure is inconsistent")
    return expanded


__all__ = [
    "HRN",
    "Subschema",
    "ValidationReport",
    "hand_written_subschema",
    "owlrl_closure",
    "schema_shapes",
    "validate_subschema",
]
