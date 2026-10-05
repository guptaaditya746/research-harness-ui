"""The provenance graph of one run: DBLP seed facts and extracted triples as RDF events.

Used by ``RunOutcomes._success``. Seed facts become one ``corpus-seed`` event; each extracted
triple that passes its checks becomes an ``extract-<n>`` event and a UI ``Triple``.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

from harness_extraction.claims import valid_edge
from harness_graphstore import ProvenanceDataset
from harness_schema.models import Triple
from harness_semantic import DBLP, HRN, Subschema
from rdflib import RDFS, XSD, Literal, URIRef

# Harness predicates that are written with their DBLP equivalent in the RDF export.
DBLP_PREDICATES = {
    "title": DBLP.title,
    "authoredBy": DBLP.authoredBy,
    "year": DBLP.yearOfPublication,
    "venue": DBLP.publishedIn,
}


def resource(value: str) -> URIRef:
    return URIRef(f"https://harness.local/resource/{quote(value, safe='')}")


def _predicate(schema: Subschema, name: str) -> tuple[URIRef, Any]:
    """The exported predicate and the range the schema expects for it."""
    local = URIRef(HRN[name])
    predicate = URIRef(DBLP_PREDICATES.get(name, local))
    return predicate, schema.graph.value(predicate, RDFS.range) or schema.graph.value(local, RDFS.range)


def _is_literal_range(expected: Any) -> bool:
    return bool(expected) and str(expected).startswith(str(XSD))


def seed_rdf(corpus: Any, schema: Subschema, rows: list[dict[str, Any]]) -> list[tuple[Any, URIRef, Any]]:
    """Seed fact rows as RDF; a citation of a paper outside DBLP becomes ``citesTitle`` with its title."""
    rdf: list[tuple[Any, URIRef, Any]] = []
    for row in rows:
        raw_predicate = str(row["predicate"])
        name = {"author": "authoredBy"}.get(raw_predicate, raw_predicate)
        title = None
        if raw_predicate == "cites" and not corpus.canonical_paper_iri(str(row.get("object", ""))):
            title = corpus.reference_title(str(row.get("source_id", "")))
            if not title:
                continue
            name = "citesTitle"
        if not hasattr(HRN, name):
            continue
        predicate, expected = _predicate(schema, name)
        subject_value = corpus.canonical_paper_iri(str(row["subject"])) or str(row["subject"])
        subject = URIRef(subject_value) if subject_value.startswith("https://dblp.org/rec/") else resource(subject_value)
        raw = title if name == "citesTitle" else row.get("object")
        if name == "citesTitle" or _is_literal_range(expected):
            object_: Any = Literal(raw)
        else:
            object_ = URIRef(corpus.canonical_paper_iri(str(raw)) or resource(str(raw)))
        rdf.append((subject, predicate, object_))
    return rdf


def add_extracted(dataset: ProvenanceDataset, corpus: Any, schema: Subschema,
                  extracted: list[Any]) -> tuple[list[Triple], dict[str, int]]:
    """Add each extracted triple as its own event; returns the UI triples and why others were dropped."""
    triples: list[Triple] = []
    dropped = {"invalid edge": 0, "paper not indexed": 0}
    for index, item in enumerate(extracted, start=1):
        candidate = Triple(s=item.subject, p=item.predicate, o=item.object,
                           support="1 passage", evidence=item.evidence, source=item.source_id)
        if not valid_edge(candidate):
            dropped["invalid edge"] += 1
            continue
        predicate, expected = _predicate(schema, item.predicate)
        subject_iri = corpus.canonical_paper_iri(item.subject)
        object_iri = corpus.canonical_paper_iri(item.object)
        if expected and str(expected) == str(HRN.Paper) and not object_iri and item.object not in corpus.by_paper:
            # A paper-valued relation must resolve to an indexed paper.
            dropped["paper not indexed"] += 1
            continue
        object_ = (Literal(item.object) if _is_literal_range(expected)
                   else URIRef(object_iri) if object_iri else resource(item.object))
        subject = URIRef(subject_iri) if subject_iri else resource(item.subject)
        dataset.add_event(f"extract-{index}", [(subject, predicate, object_)], [item.source_id])
        triples.append(Triple(
            s=subject_iri or item.subject, p=item.predicate, o=object_iri or item.object,
            support="1 passage", evidence=item.evidence, source=item.source_id, tripleId=f"t-{index:03d}",
        ))
    return triples, dropped
