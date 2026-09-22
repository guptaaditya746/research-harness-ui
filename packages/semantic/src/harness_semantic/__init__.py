"""M2 hand-written schema and its static SHACL constraints."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from owlrl import DeductiveClosure, OWLRL_Semantics
from pyshacl import validate
from rdflib import OWL, RDF, RDFS, SH, XSD, Graph, Literal, Namespace, URIRef

HRN = Namespace("https://harness.local/schema#")
DBLP = Namespace("https://dblp.org/rdf/schema#")


@dataclass(frozen=True)
class Subschema:
    graph: Graph
    classes: frozenset[URIRef]
    properties: frozenset[URIRef]
    structural_properties: frozenset[URIRef]
    dblp_graph: Graph | None = None
    seed_schema_version: str = "hand-written"
    seed_schema_sha256: str | None = None


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


def _default_dblp_schema_path() -> Path:
    configured = os.getenv("DBLP_SCHEMA_PATH", "").strip()
    if configured:
        return Path(configured)
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "seed" / "dblp_schema_2024-06-14.ttl"
        if candidate.exists():
            return candidate
    return Path("seed/dblp_schema_2024-06-14.ttl")


def _load_dblp_seed(path: Path | None = None) -> tuple[Graph | None, str, str | None]:
    seed_path = path or _default_dblp_schema_path()
    if not seed_path.exists():
        return None, "hand-written", None
    graph = Graph()
    graph.parse(seed_path, format="turtle")
    digest = hashlib.sha256(seed_path.read_bytes()).hexdigest()
    return graph, "2024-06-14", digest


def hand_written_subschema(schema_path: Path | None = None) -> Subschema:
    """Return the compatibility schema enriched with the immutable DBLP seed."""
    graph = Graph()
    graph.bind("hrn", HRN)
    graph.bind("dblp", DBLP)
    graph.bind("xsd", XSD)
    dblp_graph, seed_version, seed_hash = _load_dblp_seed(schema_path)
    if dblp_graph is not None:
        for triple in dblp_graph:
            graph.add(triple)
    classes = frozenset({HRN.Paper, HRN.Author, HRN.Venue, HRN.Dataset, HRN.Metric, HRN.Claim})
    for iri in classes:
        _add_class(graph, iri)
    graph.add((HRN.Paper, RDFS.subClassOf, DBLP.Publication))
    graph.add((HRN.Author, RDFS.subClassOf, DBLP.Creator))

    specs = {
        HRN.title: (HRN.Paper, XSD.string),
        HRN.authoredBy: (HRN.Paper, HRN.Author),
        HRN.year: (HRN.Paper, XSD.gYear),
        HRN.venue: (HRN.Paper, HRN.Venue),
        HRN.cites: (HRN.Paper, HRN.Paper),
        HRN.citesTitle: (HRN.Paper, XSD.string),
        HRN.sectionOf: (HRN.Claim, HRN.Paper),
        HRN.hasTable: (HRN.Paper, XSD.string),
        HRN.usesDataset: (HRN.Paper, HRN.Dataset),
        HRN.reportsMetric: (HRN.Paper, HRN.Metric),
        HRN.claims: (HRN.Paper, HRN.Claim),
        HRN.comparesTo: (HRN.Paper, HRN.Paper),
    }
    for iri, (domain, range_) in specs.items():
        _add_property(graph, iri, domain, range_)
    equivalents = {
        HRN.title: DBLP.title,
        HRN.authoredBy: DBLP.authoredBy,
        HRN.year: DBLP.yearOfPublication,
        HRN.venue: DBLP.publishedIn,
    }
    for local, official in equivalents.items():
        graph.add((local, OWL.equivalentProperty, official))
    structural = frozenset(
        {HRN.title, HRN.authoredBy, HRN.year, HRN.venue, HRN.cites, HRN.citesTitle, HRN.sectionOf, HRN.hasTable}
    )
    return Subschema(graph, classes, frozenset(specs), structural, dblp_graph, seed_version, seed_hash)


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
    "DBLP",
    "HRN",
    "Subschema",
    "ValidationReport",
    "hand_written_subschema",
    "owlrl_closure",
    "schema_shapes",
    "validate_subschema",
]
