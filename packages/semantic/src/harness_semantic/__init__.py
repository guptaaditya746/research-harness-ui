"""M2 hand-written schema and its static SHACL constraints."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from harness_config import settings
from owlrl import DeductiveClosure, OWLRL_Semantics
from pyshacl import validate
from rdflib import OWL, RDF, RDFS, SH, XSD, Graph, Literal, Namespace, URIRef

HRN = Namespace("https://harness.local/schema#")
DBLP = Namespace("https://dblp.org/rdf/schema#")
CSKG = Namespace("https://w3id.org/cskg/ontology#")
CSKG_SUBSET_FILE = "cskg_ontology_2.0_subset.ttl"
CSKG_VERSION = "2.0-subset"

# Parents for invented terms (Phase 4): every invented property specialises one of these (or a
# more specific core/DBLP property), every invented class specialises an existing class.
TOP_RELATION = HRN.researchRelation     # owl:ObjectProperty: a relation between research entities
TOP_ATTRIBUTE = HRN.researchAttribute   # owl:DatatypeProperty: a literal value of a research entity
TOP_CLASS = HRN.ResearchEntity          # owl:Class: anything a paper studies, uses or reports
# Harness terms declared equivalent to DBLP properties: they count as DBLP vocabulary.
DBLP_EQUIVALENTS = {
    HRN.title: DBLP.title,
    HRN.authoredBy: DBLP.authoredBy,
    HRN.year: DBLP.yearOfPublication,
    HRN.venue: DBLP.publishedIn,
}
# The content core (Phase 4): CS-KG 2.0 research-entity classes and relations,
# named by harness terms declared equivalent to them. Relations between research entities; what a
# paper does is expressed with the paper-level properties (addresses, evaluatesOn, usesDataset, ...).
CSKG_CLASSES = {                      # harness class -> CS-KG class (hrn:Metric already existed)
    HRN.Method: CSKG.Method,
    HRN.Task: CSKG.Task,
    HRN.Material: CSKG.Material,
    HRN.Metric: CSKG.Metric,
    HRN.OtherEntity: CSKG.OtherEntity,
}
# harness relation -> (CS-KG relation, its inverse in CS-KG, range, definition shown to the models).
CSKG_RELATIONS = {
    HRN.uses: (CSKG.uses, CSKG.usedBy, TOP_CLASS,
               "A method, task or other research entity employs, applies or builds on another one."),
    HRN.solves: (CSKG.solves, CSKG.solvedBy, HRN.Task, "A method (or task) solves or tackles a task."),
    HRN.improves: (CSKG.improves, CSKG.improvedBy, TOP_CLASS,
                   "A research entity makes another one better (a method, a metric, a task's results)."),
    HRN.outperforms: (CSKG.outperforms, CSKG.outperformedBy, TOP_CLASS,
                      "A method or system performs better than another one."),
    HRN.includes: (CSKG.includes, CSKG.includedBy, TOP_CLASS,
                   "A research entity has another one as a part or component."),
    HRN.produces: (CSKG.produces, CSKG.producedBy, TOP_CLASS,
                   "A research entity generates or yields another one (an output, a resource)."),
    HRN.supports: (CSKG.supports, CSKG.supportedBy, TOP_CLASS,
                   "A research entity aids or enables another one."),
    HRN.analyzes: (CSKG.analyzes, CSKG.analyzedBy, TOP_CLASS,
                   "A method or task examines or studies another research entity."),
}
# Inverse names an agent may write ("usedBy") -> the harness relation, read the other way round.
CSKG_INVERSES = {str(inverse).rsplit("#", 1)[-1]: str(local).rsplit("#", 1)[-1]
                 for local, (_, inverse, _, _) in CSKG_RELATIONS.items()}
CSKG_EQUIVALENTS = {**CSKG_CLASSES, **{local: official for local, (official, *_) in CSKG_RELATIONS.items()}}
# Fixed paper-level properties of the content core: one name and one type for each, in every run.
PAPER_CORE = {
    HRN.addresses: (HRN.Task, TOP_RELATION, "The paper works on the task: the problem it sets out to solve or study."),
    HRN.evaluatesOn: (HRN.Dataset, HRN.usesDataset,
                      "The paper evaluates its methods or models on the dataset or benchmark."),
}
ORIGINS = ("dblp", "core", "invented")


@dataclass(frozen=True)
class Subschema:
    graph: Graph
    classes: frozenset[URIRef]
    properties: frozenset[URIRef]
    structural_properties: frozenset[URIRef]
    dblp_graph: Graph | None = None
    seed_schema_version: str = "hand-written"
    seed_schema_sha256: str | None = None
    content_core: str | None = None          # "cskg-2.0-subset": the CS-KG content core that was loaded
    content_core_sha256: str | None = None


@dataclass(frozen=True)
class ValidationReport:
    conforms: bool
    graph: Graph
    text: str


def is_datatype_range(range_: object) -> bool:
    """A literal range (XSD datatype or rdfs:Literal) makes a datatype property; anything else an object one."""
    return str(range_).startswith(str(XSD)) or range_ == RDFS.Literal


def property_kind(graph: Graph, iri: URIRef) -> str | None:
    """"object" or "datatype" from the OWL 2 declaration, else from the range; None when unknown."""
    if (iri, RDF.type, OWL.DatatypeProperty) in graph:
        return "datatype"
    if (iri, RDF.type, OWL.ObjectProperty) in graph:
        return "object"
    range_ = graph.value(iri, RDFS.range)
    return None if range_ is None else ("datatype" if is_datatype_range(range_) else "object")


def _add_class(graph: Graph, iri: URIRef, parent: URIRef | None = None) -> None:
    graph.add((iri, RDF.type, OWL.Class))
    graph.add((iri, RDFS.label, Literal(str(iri).rsplit("#", 1)[-1])))
    if parent is not None:
        graph.add((iri, RDFS.subClassOf, parent))


def _add_property(graph: Graph, iri: URIRef, domain: URIRef, range_: URIRef, parent: URIRef | None = None) -> None:
    """Declare an OWL 2 object or datatype property (by its range), with one domain and one range."""
    graph.add((iri, RDF.type, OWL.DatatypeProperty if is_datatype_range(range_) else OWL.ObjectProperty))
    graph.add((iri, RDFS.label, Literal(str(iri).rsplit("#", 1)[-1])))
    graph.add((iri, RDFS.domain, domain))
    graph.add((iri, RDFS.range, range_))
    if parent is not None:
        graph.add((iri, RDFS.subPropertyOf, parent))


def _declare_dblp(graph: Graph) -> None:
    """OWL 2 declarations for the DBLP seed, which types its properties only as rdf:Property.

    The seed file stays untouched; the declarations are added to the merged graph so every
    property in a subschema is an owl:ObjectProperty or owl:DatatypeProperty.
    """
    for prop in set(graph.subjects(RDF.type, RDF.Property)):
        if not str(prop).startswith(str(DBLP)):
            continue
        kind = property_kind(graph, prop)
        if kind is not None:
            graph.add((prop, RDF.type, OWL.DatatypeProperty if kind == "datatype" else OWL.ObjectProperty))


def _default_dblp_schema_path() -> Path:
    configured = settings().dblp_schema_path.strip()
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


def cskg_subset_path() -> Path:
    """The CS-KG subset file: CSKG_SUBSET_PATH, else seed/cskg_ontology_2.0_subset.ttl in the checkout."""
    configured = settings().cskg_subset_path.strip()
    if configured:
        return Path(configured).expanduser()
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "seed" / CSKG_SUBSET_FILE
        if candidate.exists():
            return candidate
    return Path("seed") / CSKG_SUBSET_FILE


def _add_content_core(graph: Graph) -> tuple[frozenset[URIRef], frozenset[URIRef], str]:
    """The CS-KG subset plus the harness terms declared equivalent to it and the paper-level core.
    Returns the harness classes and properties it adds and the subset file's SHA-256."""
    path = cskg_subset_path()
    if not path.exists():
        raise FileNotFoundError(f"CS-KG subset not found: {path} (set CSKG_SUBSET_PATH)")
    graph.parse(path, format="turtle")
    graph.bind("cskg-ont", CSKG)
    graph.add((CSKG.ResearchEntity, RDFS.subClassOf, TOP_CLASS))
    for local, official in CSKG_CLASSES.items():
        if (official, RDF.type, OWL.Class) not in graph:
            raise ValueError(f"{path} does not declare {official}")
        _add_class(graph, local, TOP_CLASS)
        graph.add((local, OWL.equivalentClass, official))
        comment = graph.value(official, RDFS.comment)
        if comment is not None and graph.value(local, RDFS.comment) is None:
            graph.add((local, RDFS.comment, Literal(str(comment))))   # CS-KG's definition, unchanged
    graph.remove((HRN.Dataset, RDFS.subClassOf, TOP_CLASS))
    graph.add((HRN.Dataset, RDFS.subClassOf, HRN.Material))
    graph.add((HRN.Dataset, RDFS.comment, Literal(
        "A collection of data that a paper trains, tests or evaluates on (a kind of CS-KG Material).")))
    for local, (official, _inverse, range_, definition) in CSKG_RELATIONS.items():
        if (official, RDF.type, OWL.ObjectProperty) not in graph:
            raise ValueError(f"{path} does not declare {official}")
        _add_property(graph, local, TOP_CLASS, range_, TOP_RELATION)
        graph.add((local, OWL.equivalentProperty, official))
        graph.add((local, RDFS.comment, Literal(definition)))
    for local, (range_, parent, definition) in PAPER_CORE.items():
        _add_property(graph, local, HRN.Paper, range_, parent)
        graph.add((local, RDFS.comment, Literal(definition)))
    properties = frozenset({*CSKG_RELATIONS, *PAPER_CORE})
    return frozenset(CSKG_CLASSES), properties, hashlib.sha256(path.read_bytes()).hexdigest()


def hand_written_subschema(schema_path: Path | None = None) -> Subschema:
    """Return the compatibility schema enriched with the immutable DBLP seed and the CS-KG content core."""
    graph = Graph()
    graph.bind("hrn", HRN)
    graph.bind("dblp", DBLP)
    graph.bind("xsd", XSD)
    graph.bind("owl", OWL)
    dblp_graph, seed_version, seed_hash = _load_dblp_seed(schema_path)
    if dblp_graph is not None:
        for triple in dblp_graph:
            graph.add(triple)
        _declare_dblp(graph)
    classes = frozenset({HRN.Paper, HRN.Author, HRN.Venue, HRN.Dataset, HRN.Metric, HRN.Claim})
    _add_class(graph, TOP_CLASS)
    graph.add((TOP_CLASS, RDFS.comment, Literal(
        "Anything a paper studies, uses, proposes or reports; the parent of invented classes.")))
    for iri in classes:
        _add_class(graph, iri, TOP_CLASS if iri in {HRN.Dataset, HRN.Metric, HRN.Claim} else None)
    graph.add((HRN.Paper, RDFS.subClassOf, DBLP.Publication))
    graph.add((HRN.Author, RDFS.subClassOf, DBLP.Creator))
    graph.add((HRN.Venue, RDFS.subClassOf, DBLP.Stream))
    _add_property(graph, TOP_RELATION, OWL.Thing, OWL.Thing)
    graph.add((TOP_RELATION, RDFS.comment, Literal(
        "A relation a paper or research entity has to another entity; the parent of invented object properties.")))
    _add_property(graph, TOP_ATTRIBUTE, OWL.Thing, RDFS.Literal)
    graph.add((TOP_ATTRIBUTE, RDFS.comment, Literal(
        "A literal value (number, name, phrase) of a paper or research entity; the parent of invented "
        "datatype properties.")))

    specs = {
        HRN.title: (HRN.Paper, XSD.string),
        HRN.authoredBy: (HRN.Paper, HRN.Author),
        HRN.year: (HRN.Paper, XSD.gYear),
        # Equivalent to dblp:publishedIn (a string), so a datatype property too (OWL 2 DL: one kind).
        HRN.venue: (HRN.Paper, XSD.string),
        HRN.cites: (HRN.Paper, HRN.Paper),
        HRN.citesTitle: (HRN.Paper, XSD.string),
        HRN.sectionOf: (HRN.Claim, HRN.Paper),
        HRN.hasTable: (HRN.Paper, XSD.string),
        HRN.usesDataset: (HRN.Paper, HRN.Dataset),
        HRN.reportsMetric: (HRN.Paper, HRN.Metric),
        HRN.claims: (HRN.Paper, HRN.Claim),
        HRN.comparesTo: (HRN.Paper, HRN.Paper),
    }
    research = {HRN.usesDataset, HRN.reportsMetric, HRN.claims, HRN.comparesTo, HRN.hasTable}
    for iri, (domain, range_) in specs.items():
        parent = (TOP_ATTRIBUTE if is_datatype_range(range_) else TOP_RELATION) if iri in research else None
        _add_property(graph, iri, domain, range_, parent)
    for local, official in DBLP_EQUIVALENTS.items():
        graph.add((local, OWL.equivalentProperty, official))
    structural = frozenset(
        {HRN.title, HRN.authoredBy, HRN.year, HRN.venue, HRN.cites, HRN.citesTitle, HRN.sectionOf, HRN.hasTable}
    )
    properties = frozenset(specs)
    core_classes, core_properties, core_hash = _add_content_core(graph)
    classes, properties, core = classes | core_classes, properties | core_properties, f"cskg-{CSKG_VERSION}"
    return Subschema(graph, classes | {TOP_CLASS}, properties, structural, dblp_graph, seed_version, seed_hash,
                     core, core_hash)


CORE_TERMS = frozenset({
    HRN.Paper, HRN.Author, HRN.Venue, HRN.Dataset, HRN.Metric, HRN.Claim, TOP_CLASS,
    HRN.title, HRN.authoredBy, HRN.year, HRN.venue, HRN.cites, HRN.citesTitle, HRN.sectionOf, HRN.hasTable,
    HRN.usesDataset, HRN.reportsMetric, HRN.claims, HRN.comparesTo, TOP_RELATION, TOP_ATTRIBUTE,
})
# Added by the CS-KG content core.
CONTENT_CORE_TERMS = frozenset({*CSKG_CLASSES, *CSKG_RELATIONS, *PAPER_CORE})


def core_terms() -> frozenset[URIRef]:
    """The harness seed terms: CORE_TERMS plus the content core."""
    return CORE_TERMS | CONTENT_CORE_TERMS


def term_origin(iri: object) -> str:
    """"dblp" (a DBLP term or a harness term declared equivalent to one), "core" (the harness seed,
    including the CS-KG content core) or "invented" (added for one question)."""
    text = str(iri)
    if text.startswith(str(DBLP)) or URIRef(text) in DBLP_EQUIVALENTS:
        return "dblp"
    if URIRef(text) in core_terms() or text.startswith(str(CSKG)):
        return "core"
    return "invented"


def schema_shapes() -> Graph:
    shapes = Graph()
    shape = HRN.PropertyShape
    shapes.add((shape, RDF.type, SH.NodeShape))
    for target in (RDF.Property, OWL.ObjectProperty, OWL.DatatypeProperty):
        shapes.add((shape, SH.targetClass, target))
    for index, predicate in enumerate((RDFS.domain, RDFS.range), start=1):
        constraint = URIRef(f"{shape}/constraint/{index}")
        shapes.add((shape, SH.property, constraint))
        shapes.add((constraint, SH.path, predicate))
        shapes.add((constraint, SH.minCount, Literal(1)))
        shapes.add((constraint, SH.maxCount, Literal(1)))
    return shapes


def validate_subschema(graph: Graph) -> ValidationReport:
    """Every property the harness declares has one domain and one range. CS-KG's own terms are left
    out: its base relations carry no domain or range (CS-KG puts those on per-target variants such as
    usesMaterial); the harness terms equivalent to them are checked."""
    data = Graph()
    for triple in graph:
        if not str(triple[0]).startswith(str(CSKG)):
            data.add(triple)
    conforms, report_graph, report_text = validate(
        data_graph=data,
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
    "CONTENT_CORE_TERMS",
    "CORE_TERMS",
    "CSKG",
    "CSKG_CLASSES",
    "CSKG_EQUIVALENTS",
    "CSKG_INVERSES",
    "CSKG_RELATIONS",
    "CSKG_SUBSET_FILE",
    "DBLP",
    "DBLP_EQUIVALENTS",
    "HRN",
    "ORIGINS",
    "PAPER_CORE",
    "TOP_ATTRIBUTE",
    "TOP_CLASS",
    "TOP_RELATION",
    "Subschema",
    "ValidationReport",
    "core_terms",
    "cskg_subset_path",
    "hand_written_subschema",
    "is_datatype_range",
    "owlrl_closure",
    "property_kind",
    "schema_shapes",
    "term_origin",
    "validate_subschema",
]
