"""Per-run RDF dataset, provenance, SPARQL checks, and query coverage plans."""

from __future__ import annotations

import re
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from harness_ledger import Ledger, Obligation
from harness_semantic import HRN, Subschema
from rdflib import OWL, PROV, RDF, RDFS, Dataset, Graph, Literal, URIRef
from rdflib.plugins.sparql import prepareQuery
from rdflib.plugins.sparql.parserutils import CompValue
from rdflib.term import Node

_SPARQL_PARSE_LOCK = threading.Lock()


class InvariantViolation(ValueError):
    def __init__(self, number: int, message: str) -> None:
        self.number = number
        super().__init__(f"invariant:{number}: {message}")


@dataclass(frozen=True)
class CoveragePlan:
    shape: str
    properties: tuple[str, ...]
    domain: tuple[str, ...]
    obligations: list[Obligation]


def _source_iri(source_id: str) -> URIRef:
    return URIRef(f"urn:harness:source:{quote(source_id, safe='')}")


def _source_id(source_iri: URIRef) -> str:
    from urllib.parse import unquote

    return unquote(str(source_iri).rsplit(":", 1)[-1])


def _schema_range(schema: Subschema, predicate: URIRef) -> URIRef | None:
    value = schema.graph.value(predicate, RDFS.range)
    return value if isinstance(value, URIRef) else None


class ProvenanceDataset:
    def __init__(self, schema: Subschema, corpus_source_ids: set[str]) -> None:
        self.schema = schema
        self.corpus_source_ids = corpus_source_ids
        self.dataset = Dataset(default_union=True)
        self.dataset.bind("hrn", HRN)
        self.dataset.bind("prov", PROV)

    def add_event(
        self,
        event_id: str,
        triples: Iterable[tuple[Node, URIRef, Node]],
        source_ids: list[str],
    ) -> URIRef:
        if not event_id or not source_ids:
            raise InvariantViolation(2, "an extraction event requires an id and provenance")
        missing = set(source_ids) - self.corpus_source_ids
        if missing:
            raise InvariantViolation(2, f"unknown provenance sources: {sorted(missing)}")
        graph_iri = URIRef(f"urn:harness:graph:{quote(event_id, safe='')}")
        checked = list(triples)
        for _subject, predicate, object_ in checked:
            if not _known_predicate(self.schema, predicate) and predicate != RDF.type:
                raise InvariantViolation(3, f"predicate outside subschema: {predicate}")
            if predicate == RDF.type and object_ not in self.schema.classes:
                raise InvariantViolation(3, f"class outside subschema: {object_}")
            expected = _schema_range(self.schema, predicate)
            if expected is not None:
                if str(expected).startswith("http://www.w3.org/2001/XMLSchema#"):
                    if not isinstance(object_, Literal):
                        raise ValueError(f"{predicate} requires a literal value")
                elif not isinstance(object_, URIRef):
                    raise ValueError(f"{predicate} requires an IRI value")
        graph = self.dataset.graph(graph_iri)
        for triple in checked:
            graph.add(triple)
        metadata = self.dataset.default_graph
        for source_id in source_ids:
            metadata.add((graph_iri, PROV.wasDerivedFrom, _source_iri(source_id)))
        return graph_iri

    def _supporting_graphs(self, triple: tuple[Node, URIRef, Node]) -> list[Graph]:
        return [
            graph
            for graph in self.dataset.graphs()
            if graph.identifier != self.dataset.default_graph.identifier and triple in graph
        ]

    def support_count(self, triple: tuple[Node, URIRef, Node]) -> int:
        return len(self._supporting_graphs(triple))

    def provenance(self, triple: tuple[Node, URIRef, Node]) -> set[str]:
        sources: set[str] = set()
        for graph in self._supporting_graphs(triple):
            for source in self.dataset.default_graph.objects(graph.identifier, PROV.wasDerivedFrom):
                if isinstance(source, URIRef):
                    sources.add(_source_id(source))
        return sources

    def export_trig(self) -> str:
        value = self.dataset.serialize(format="trig")
        return value.decode() if isinstance(value, bytes) else value

    def export_turtle(self) -> str:
        union = Graph()
        union.bind("hrn", HRN)
        for graph in self.dataset.graphs():
            if graph.identifier == self.dataset.default_graph.identifier:
                continue
            for triple in graph:
                union.add(triple)
        value = union.serialize(format="turtle")
        return value.decode() if isinstance(value, bytes) else value

    def query(self, query: str, ledger: Ledger, coverage_statement: str):
        if not ledger.queryable:
            raise InvariantViolation(5, "query attempted while obligations remain open")
        if not coverage_statement.strip():
            raise InvariantViolation(5, "answer must state ledger coverage")
        return self.dataset.query(query)


def _walk(value: Any) -> Iterable[Any]:
    yield value
    if isinstance(value, (CompValue, dict)):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, (list, tuple, set)):
        for child in value:
            yield from _walk(child)
    elif hasattr(value, "args"):
        yield from _walk(value.args)


def _query_patterns(query: str) -> list[tuple[Any, Any, Any]]:
    # rdflib/pyparsing shares parser state across calls. Concurrent run threads
    # otherwise corrupt its parse actions and reject valid queries.
    with _SPARQL_PARSE_LOCK:
        prepared = prepareQuery(query)
    patterns: list[tuple[Any, Any, Any]] = []
    for value in _walk(prepared.algebra):
        if isinstance(value, CompValue) and value.name == "BGP":
            patterns.extend(value.get("triples", []))
    return patterns


def _query_terms(query: str) -> tuple[set[URIRef], set[URIRef]]:
    predicates: set[URIRef] = set()
    classes: set[URIRef] = set()
    for _subject, predicate, object_ in _query_patterns(query):
        path_iris = {item for item in _walk(predicate) if isinstance(item, URIRef)}
        predicates.update(path_iris)
        if predicate == RDF.type and isinstance(object_, URIRef):
            classes.add(object_)
    return predicates, classes


def _known_predicate(schema: Subschema, predicate: URIRef) -> bool:
    """Accept immutable seed predicates linked to local compatibility terms."""
    if predicate in schema.properties:
        return True
    return any(
        True
        for local in schema.properties
        for _ in schema.graph.triples((local, OWL.equivalentProperty, predicate))
    ) or any(
        True
        for local in schema.properties
        for _ in schema.graph.triples((predicate, OWL.equivalentProperty, local))
    )


def sparql_check(query: str, schema: Subschema) -> str:
    try:
        predicates, classes = _query_terms(query)
    except Exception as error:
        raise ValueError(f"invalid SPARQL: {error}") from error
    unknown_predicates = {
        predicate
        for predicate in predicates
        if predicate != RDF.type and not _known_predicate(schema, predicate)
    }
    unknown_classes = classes - schema.classes
    if unknown_predicates or unknown_classes:
        values = sorted(map(str, unknown_predicates | unknown_classes))
        raise InvariantViolation(3, f"terms outside subschema: {values}")
    prefixes = {
        "hrn": "https://harness.local/schema#",
        "rdf": str(RDF),
        "xsd": "http://www.w3.org/2001/XMLSchema#",
    }
    missing = [
        f"PREFIX {name}: <{iri}>"
        for name, iri in prefixes.items()
        if not re.search(rf"(?im)^\s*PREFIX\s+{name}:", query)
    ]
    return "\n".join([*missing, query])


def _property_kind(schema: Subschema, iri: URIRef) -> str:
    return "structural" if iri == RDF.type or iri in schema.structural_properties else "content"


def analyse_query(query: str, schema: Subschema, facts: list[dict[str, object]]) -> CoveragePlan:
    sparql_check(query, schema)
    upper = query.upper()
    if re.search(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(|\bORDER\s+BY\b|\bGROUP\s+BY\b", upper):
        shape = "aggregate"
    elif re.search(r"(?:<[^>]+>|[A-Za-z_][\w-]*:[\w-]+)\s*[+*]", query):
        shape = "path"
    elif re.search(r"\{\s*<[^>]+>\s+", query) or re.search(r"\bFILTER\s*\(", upper):
        shape = "point"
    else:
        shape = "set"

    properties, classes = _query_terms(query)
    relevant = sorted(properties - {RDF.type}, key=str)
    if not relevant and classes:
        relevant = [RDF.type]
    expected_objects = {
        predicate: str(object_)
        for _subject, predicate, object_ in _query_patterns(query)
        if isinstance(predicate, URIRef) and isinstance(object_, URIRef)
    }
    domain_ids = sorted({str(fact["subject"]) for fact in facts if fact.get("subject")})
    if shape == "point":
        match = re.search(r"\{\s*<([^>]+)>\s+", query)
        domain_ids = [match.group(1)] if match else domain_ids[:1]
    obligations = [
        Obligation(
            id=f"o-{index:03d}",
            subject=subject if "://" in subject else f"https://harness.local/resource/{subject}",
            property=str(prop),
            kind=_property_kind(schema, prop),  # type: ignore[arg-type]
            expected_object=expected_objects.get(prop),
        )
        for index, (subject, prop) in enumerate(
            ((subject, prop) for subject in domain_ids for prop in relevant), start=1
        )
    ]
    return CoveragePlan(shape, tuple(map(str, relevant)), tuple(domain_ids), obligations)


def check_answer_claims(
    claims: list[dict[str, object]],
    result_row_ids: set[str],
    triple_sources: dict[str, set[str]],
    passage_ids: set[str],
) -> None:
    for claim in claims:
        rows = set(claim.get("row_ids", []))
        triples = set(claim.get("triple_ids", []))
        if not rows or not rows <= result_row_ids:
            raise InvariantViolation(4, f"claim has no query-result row: {claim.get('text', '')}")
        if not triples:
            raise InvariantViolation(4, f"claim has no cited triple: {claim.get('text', '')}")
        for triple_id in triples:
            sources = triple_sources.get(str(triple_id), set())
            if not sources or not sources <= passage_ids:
                raise InvariantViolation(
                    4, f"cited triple does not resolve to a passage: {triple_id}"
                )


__all__ = [
    "CoveragePlan",
    "InvariantViolation",
    "ProvenanceDataset",
    "analyse_query",
    "check_answer_claims",
    "sparql_check",
]
