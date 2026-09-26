"""The DBLP backbone of an answer and the origin of its schema terms (Phase 4).

Research answers are extracted from passages, so on their own they hold no DBLP facts and their
papers rarely share a node. The backbone adds, for the papers an answer uses, their authors, their
venue (as one shared node per venue, the DBLP ``publishedInStream`` sense) and the citations
among them, all from verified corpus facts. It goes into the answer graph only: it is context, not
something the run extracted, so it is not part of ``Reading.triples`` (which is what gold scoring
reads). Years are left out of the graph on purpose: a shared year node would link unrelated papers.

``schema_terms`` lists a reading's subschema with each term's origin (DBLP / core / invented) for
the Subschema panel; ``subschema_slice`` exports only the terms a question uses, with their parents,
instead of the whole seed ontology.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from harness_schema.models import SchemaTerm, Triple
from harness_semantic import DBLP, DBLP_EQUIVALENTS, HRN, property_kind, term_origin
from rdflib import OWL, RDF, RDFS, XSD, Graph, URIRef

BACKBONE_PAPERS = 25          # at most this many papers get backbone edges
BACKBONE_AUTHORS = 8          # per paper
BACKBONE_PREFIX = "b-"        # edge ids of backbone facts in the answer graph
PREFIXES = {"hrn": str(HRN), "dblp": str(DBLP), "xsd": str(XSD), "owl": str(OWL), "rdfs": str(RDFS)}
FOLLOW = {RDFS.subClassOf, RDFS.subPropertyOf, OWL.equivalentProperty, RDFS.domain, RDFS.range}


def prefixed(iri: object) -> str:
    """A prefixed name for known namespaces, else the IRI in angle brackets (valid Turtle either way)."""
    text = str(iri)
    for prefix, namespace in PREFIXES.items():
        if text.startswith(namespace):
            return f"{prefix}:{text[len(namespace):]}"
    return f"<{text}>"


def _parent(graph: Graph, iri: URIRef, predicate: URIRef) -> URIRef | None:
    """One parent, preferring harness/DBLP ones (DBLP terms often have several external parents)."""
    parents = sorted(graph.objects(iri, predicate), key=str)
    known = [item for item in parents if str(item).startswith((str(HRN), str(DBLP)))]
    return (known or parents or [None])[0]


def _paper_key(corpus: Any, value: str) -> str | None:
    try:
        return corpus.paper_key(value)
    except Exception:  # noqa: BLE001 - UnknownIdentifier or a value that is not a paper
        return None


def answer_paper_keys(corpus: Any, subjects: Iterable[str], objects: Iterable[str] = (),
                      sources: Iterable[str] = ()) -> list[str]:
    """Local keys of the papers an answer is about: subjects, objects that are papers, and the
    papers of the passages its triples come from."""
    keys: list[str] = []
    by_passage = getattr(corpus, "by_passage", {}) or {}
    candidates = [*subjects, *(value for value in objects
                               if corpus.canonical_paper_iri(value) or value in corpus.by_paper),
                  *(str(by_passage[source]["paper_id"]) for source in sources if source in by_passage)]
    for value in candidates:
        key = _paper_key(corpus, str(value))
        if key and key not in keys:
            keys.append(key)
    return keys


def backbone_triples(corpus: Any, keys: list[str]) -> list[Triple]:
    """Authors, venue and in-corpus citations among ``keys``, as graph triples with provenance."""
    chosen = keys[:BACKBONE_PAPERS]
    wanted = set(chosen)
    iri = {key: corpus.canonical_paper_iri(key) or key for key in chosen}
    authors: dict[str, int] = {}
    out: list[Triple] = []
    for row in corpus.fact_rows:
        subject = str(row.get("subject", ""))
        if subject not in wanted:
            continue
        predicate = str(row.get("predicate", ""))
        obj = str(row.get("object", "")).strip()
        source = str(row.get("source_id") or f"{subject}-metadata")
        evidence = f"Corpus fact {row.get('fact_id', source)}."
        if not obj:
            continue
        if predicate == "author":
            authors[subject] = authors.get(subject, 0) + 1
            if authors[subject] <= BACKBONE_AUTHORS:
                out.append(Triple(s=iri[subject], p="authoredBy", o=obj, support="corpus fact",
                                  evidence=evidence, source=source))
        elif predicate == "venue":
            out.append(Triple(s=iri[subject], p="publishedInStream", o=obj, support="corpus fact",
                              evidence=evidence, source=source))
        elif predicate == "cites" and row.get("in_corpus") is True and obj in wanted and obj != subject:
            out.append(Triple(s=iri[subject], p="cites", o=iri[obj], support="corpus fact",
                              evidence=evidence, source=source))
    # Own ids ("b-001"): extracted triples are "t-001"..., and the graph needs unique edge ids.
    return [triple.model_copy(update={"triple_id": f"{BACKBONE_PREFIX}{index:03d}"})
            for index, triple in enumerate(out, start=1)]


def resolve_property(graph: Graph, name: str) -> URIRef | None:
    """The IRI a predicate label stands for: a harness property, else a DBLP one ("author" = authoredBy)."""
    label = {"author": "authoredBy"}.get(name.strip(), name.strip())
    if not label:
        return None
    for candidate in (URIRef(HRN[label]), URIRef(DBLP[label])):
        if property_kind(graph, candidate) is not None or (candidate, RDF.type, RDF.Property) in graph:
            return candidate
    return None


def origin_of(graph: Graph, name: str) -> str:
    iri = resolve_property(graph, name)
    return term_origin(iri) if iri is not None else "invented"


def _term(graph: Graph, iri: URIRef, kind: str) -> SchemaTerm:
    shown = DBLP_EQUIVALENTS.get(iri, iri)          # harness equivalents are shown by their DBLP name
    relation = RDFS.subClassOf if kind == "class" else RDFS.subPropertyOf
    parent = _parent(graph, iri, relation) or _parent(graph, shown, relation)
    domain, range_ = graph.value(shown, RDFS.domain), graph.value(shown, RDFS.range)
    comment = graph.value(iri, RDFS.comment) or graph.value(shown, RDFS.comment)
    return SchemaTerm(
        name=prefixed(shown), iri=str(shown), kind=kind, origin=term_origin(iri),
        parent=prefixed(parent) if parent is not None else None,
        domain=prefixed(domain) if domain is not None else None,
        range=prefixed(range_) if range_ is not None else None,
        definition=str(comment) if comment is not None else None,
    )


def schema_terms(graph: Graph, classes: Iterable[str], properties: Iterable[str],
                 extra_properties: Iterable[str] = ()) -> list[SchemaTerm]:
    """The reading's classes and properties (plus backbone predicates) as OWL 2 terms with origins."""
    terms: list[SchemaTerm] = []
    seen: set[str] = set()
    for name in classes:
        iri = URIRef(HRN[name])
        if str(iri) not in seen:
            seen.add(str(iri))
            terms.append(_term(graph, iri, "class"))
    for name in [*properties, *extra_properties]:
        iri = resolve_property(graph, name)
        if iri is None or str(iri) in seen:
            continue
        seen.add(str(iri))
        terms.append(_term(graph, iri, property_kind(graph, DBLP_EQUIVALENTS.get(iri, iri)) or "object"))
    return terms


def subschema_slice(graph: Graph, terms: Iterable[SchemaTerm]) -> Graph:
    """Only the question's terms, their equivalents, parents, domains and ranges (HRN and DBLP)."""
    out = Graph()
    for prefix, namespace in PREFIXES.items():
        out.bind(prefix, namespace)
    todo = [URIRef(term.iri) for term in terms]
    todo += [key for key, value in DBLP_EQUIVALENTS.items() if str(value) in {term.iri for term in terms}]
    seen: set[URIRef] = set()
    while todo:
        iri = todo.pop()
        if iri in seen:
            continue
        seen.add(iri)
        for predicate, obj in graph.predicate_objects(iri):
            out.add((iri, predicate, obj))
            if predicate in FOLLOW and isinstance(obj, URIRef) and str(obj).startswith((str(HRN), str(DBLP))):
                todo.append(obj)
    return out


__all__ = ["BACKBONE_PREFIX", "answer_paper_keys", "backbone_triples", "origin_of", "prefixed", "resolve_property",
           "schema_terms", "subschema_slice"]
