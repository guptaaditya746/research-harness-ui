"""Predicate and class rules for the "entailed" matching level.

The rules are ordinary RDFS/OWL statements (W3C standards):

- ``rdfs:subPropertyOf``: ``hrn:hasAttackTask rdfs:subPropertyOf hrn:evaluatesAttack``
- ``owl:equivalentProperty``: two names for the same relation
- ``owl:inverseOf``: the same relation read in the other direction
- ``rdfs:subClassOf`` / ``owl:equivalentClass``: class hierarchy for ``rdf:type`` triples
- ``owl:sameAs``: two names for the same entity

They come from three places, merged into one graph:

1. the harness schema (``harness_semantic.hand_written_subschema``), which
   already links the harness predicates to DBLP (``hrn:authoredBy`` is
   equivalent to ``dblp:authoredBy``) and includes the DBLP seed schema;
2. ``datasets/ontology/predicate_relations.ttl``: rules for invented predicates,
   written by people (``HARNESS_PREDICATE_RELATIONS_FILE`` overrides the path);
3. nothing else: no model is involved.

The graph is closed once with the OWL 2 RL reasoner (``owlrl``), so chains
(``a ⊑ b ⊑ c``) and equivalences are resolved by a standard reasoner, not by
hand-written logic. The closed rules are then applied to scored triples one
triple at a time. Every rule here has a single premise, so the consequences of
a set of triples are the union of the consequences of each triple. That is what
lets the scorer draw an edge between one run triple and one gold triple.

Direction matters: a triple implies its generalisations, never its
specialisations. ``SafeRAG hasAttackTask "silver noise"`` implies
``SafeRAG evaluatesAttack "silver noise"``, not the reverse.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from harness_config import settings
from rdflib import OWL, RDF, RDFS, Graph, URIRef

HRN_NS = "https://harness.local/schema#"
DBLP_NS = "https://dblp.org/rdf/schema#"
# CS-KG terms are keyed by local name like harness terms: runs and gold write "uses" / "usedBy", and
# the harness names of the content core are the CS-KG local names (hrn:uses = cskg-ont:uses).
CSKG_NS = "https://w3id.org/cskg/ontology#"
LOCAL_NAMESPACES = (HRN_NS, CSKG_NS)
RDF_TYPE = str(RDF.type)
# Spellings of rdf:type that runs and people use; all mean the same predicate.
TYPE_SPELLINGS = {"type", "rdf:type", "a", "instanceof", "isa", RDF_TYPE.casefold()}

Key = tuple[str, str, str]


def _local(iri: str) -> str:
    return re.split(r"[#/]", iri.rstrip("/"))[-1]


@dataclass(frozen=True)
class PredicateRelations:
    """Closed rules, keyed by the scorer's canonical predicate and entity keys."""

    super_properties: dict[str, frozenset[str]] = field(default_factory=dict)
    inverses: dict[str, frozenset[str]] = field(default_factory=dict)
    super_classes: dict[str, frozenset[str]] = field(default_factory=dict)
    same_as: dict[str, str] = field(default_factory=dict)
    source_files: tuple[str, ...] = ()

    # ─── applying the rules ──────────────────────────────────────────────────

    @staticmethod
    def lookup(key: str) -> str:
        """Key used for class and same-as tables: no hrn: prefix, case-folded unless it is a link."""
        text = key.strip()
        if text.startswith(HRN_NS):
            text = text[len(HRN_NS):]
        elif text.startswith("hrn:"):
            text = text[4:]
        return text if text.startswith(("http://", "https://")) else " ".join(text.casefold().split())

    def canonical_entity(self, key: str) -> str:
        return self.same_as.get(self.lookup(key), key)

    def normalise(self, key: Key) -> Key:
        """One spelling for rdf:type, one name per class and per same-as group."""
        subject, predicate, object_ = key
        if predicate.casefold() in TYPE_SPELLINGS:
            predicate = RDF_TYPE
            object_ = self.lookup(object_)
        return self.canonical_entity(subject), predicate, self.canonical_entity(object_)

    def consequences(self, item: Mapping[str, Any], key: Callable[[Mapping[str, Any]], Key]) -> frozenset[Key]:
        """Every triple one triple implies under the rules, including itself, as normalised keys.

        Rules are applied to the raw subject and object and every derived triple is keyed again with
        the scorer's ``key``, because entity keys depend on the predicate (an author name is
        normalised differently as the object of ``dblp:authoredBy``).
        """
        predicate = key(item)[1]
        if predicate.casefold() in TYPE_SPELLINGS:
            predicate = RDF_TYPE
        start = (str(item.get("s", "")), predicate, str(item.get("o", "")))
        found = {start}
        frontier = [start]
        while frontier:
            subject, predicate, object_ = frontier.pop()
            derived: set[Key] = set()
            for parent in self.super_properties.get(predicate, ()):
                derived.add((subject, parent, object_))
            for inverse in self.inverses.get(predicate, ()):
                derived.add((object_, inverse, subject))
            if predicate == RDF_TYPE:
                for parent in self.super_classes.get(self.lookup(object_), ()):
                    derived.add((subject, predicate, parent))
            for triple in derived - found:
                found.add(triple)
                frontier.append(triple)
        return frozenset(self.normalise(key({"s": s, "p": p, "o": o})) for s, p, o in found)

    def related_predicates(self, predicates: Iterable[str]) -> frozenset[str]:
        """The predicates plus their parents, children and inverses."""
        wanted = set(predicates)
        related = set(wanted)
        for predicate, parents in self.super_properties.items():
            if predicate in wanted or parents & wanted:
                related.add(predicate)
                related.update(parents)
        for predicate, inverses in self.inverses.items():
            if predicate in wanted:
                related.update(inverses)
        return frozenset(related)


def default_relations_path() -> Path | None:
    configured = settings().predicate_relations_file.strip()
    if configured:
        return Path(configured).expanduser()
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "datasets" / "ontology" / "predicate_relations.ttl"
        if candidate.exists():
            return candidate
    return None


def _schema_graph() -> Graph:
    """The harness schema, including the DBLP seed (optional dependency at import time)."""
    from harness_semantic import hand_written_subschema

    return hand_written_subschema().graph


def _closed(graph: Graph) -> Graph:
    from harness_semantic import owlrl_closure

    return owlrl_closure(graph)


def build_relations(graph: Graph, *, predicate_key, entity_key, sources: tuple[str, ...] = ()) -> PredicateRelations:
    """Turn a closed RDFS/OWL graph into lookup tables in the scorer's key space."""

    def property_key(node: object) -> str | None:
        if not isinstance(node, URIRef):
            return None
        text = str(node)
        if text == RDF_TYPE:
            return RDF_TYPE
        return predicate_key(_local(text)) if text.startswith(LOCAL_NAMESPACES) else predicate_key(text)

    def class_key(node: object) -> str | None:
        if not isinstance(node, URIRef):
            return None
        text = str(node)
        return PredicateRelations.lookup(entity_key(_local(text) if text.startswith(LOCAL_NAMESPACES) else text))

    supers: dict[str, set[str]] = {}
    for child, parent in graph.subject_objects(RDFS.subPropertyOf):
        left, right = property_key(child), property_key(parent)
        if left and right and left != right:
            supers.setdefault(left, set()).add(right)
    for left_node, right_node in graph.subject_objects(OWL.equivalentProperty):
        left, right = property_key(left_node), property_key(right_node)
        if left and right and left != right:
            supers.setdefault(left, set()).add(right)
            supers.setdefault(right, set()).add(left)
    inverses: dict[str, set[str]] = {}
    for left_node, right_node in graph.subject_objects(OWL.inverseOf):
        left, right = property_key(left_node), property_key(right_node)
        if left and right:
            inverses.setdefault(left, set()).add(right)
            inverses.setdefault(right, set()).add(left)
    classes: dict[str, set[str]] = {}
    for child, parent in graph.subject_objects(RDFS.subClassOf):
        left, right = class_key(child), class_key(parent)
        if left and right and left != right:
            classes.setdefault(left, set()).add(right)
    for left_node, right_node in graph.subject_objects(OWL.equivalentClass):
        left, right = class_key(left_node), class_key(right_node)
        if left and right and left != right:
            classes.setdefault(left, set()).add(right)
            classes.setdefault(right, set()).add(left)
    groups: list[set[str]] = []
    for left_node, right_node in graph.subject_objects(OWL.sameAs):
        left, right = class_key(left_node), class_key(right_node)
        if not left or not right or left == right:
            continue
        joined = {left, right}
        for group in [group for group in groups if group & joined]:
            joined |= group
            groups.remove(group)
        groups.append(joined)
    same_as = {member: min(group) for group in groups for member in group}
    return PredicateRelations(
        super_properties={key: frozenset(value) for key, value in supers.items()},
        inverses={key: frozenset(value) for key, value in inverses.items()},
        super_classes={key: frozenset(value) for key, value in classes.items()},
        same_as=same_as,
        source_files=sources,
    )


@lru_cache(maxsize=4)
def _load_cached(path: str | None, mtime: float | None, schema_key: tuple[object, ...] = ()) -> PredicateRelations:
    """``schema_key``: what the harness schema depends on (content core setting, subset file and its
    modification time), so a changed setting is not served from the cache."""
    from . import _entity_key, _predicate_key

    graph = _schema_graph()
    sources = ["harness_semantic.hand_written_subschema"]
    if path:
        graph.parse(path, format="turtle")
        sources.append(path)
    return build_relations(
        _closed(graph),
        predicate_key=_predicate_key,
        entity_key=lambda value: _entity_key(value, predicate="", position="object"),
        sources=tuple(sources),
    )


def load_relations(path: Path | None = None) -> PredicateRelations:
    """The schema plus the predicate-relations file, closed once and cached per file version."""
    chosen = path if path is not None else default_relations_path()
    if chosen is not None and not chosen.exists():
        raise FileNotFoundError(f"predicate relations file not found: {chosen}")
    mtime = chosen.stat().st_mtime if chosen is not None else None
    return _load_cached(str(chosen) if chosen is not None else None, mtime, _schema_key())


def _schema_key() -> tuple[object, ...]:
    from harness_semantic import cskg_subset_path

    subset = cskg_subset_path()
    return ("cskg", str(subset), subset.stat().st_mtime if subset.exists() else None)


__all__ = ["CSKG_NS", "DBLP_NS", "HRN_NS", "PredicateRelations", "build_relations", "default_relations_path",
           "load_relations"]
