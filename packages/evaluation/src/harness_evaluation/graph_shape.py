"""The shape of each run's answer graph: vocabulary mix, connectivity, dead ends, schema hygiene.

Read-only diagnostics for Phase 4 (measure before enforcing anything). For one run's chosen
reading it reports:

- **Vocabulary**: fact edges whose predicate is DBLP (a ``dblp:`` property, or one of the harness
  terms declared equivalent to one: title, authoredBy, year, venue), a harness seed term (cites,
  usesDataset, reportsMetric, ...), or invented in the run.
- **Connectivity** of the knowledge part of the graph (fact, claim and derived edges; evidence
  edges and passage nodes are provenance, not knowledge; literal value nodes do not link): connected
  pieces, the largest piece's share of nodes, and the share of paper pairs connected at all.
- **Backbone**: the DBLP facts added to research answers since Phase 4 (authors, venue, citations;
  edge ids "b-..."), counted apart from what the run extracted.
- **Dead ends**: invented edges whose object no other edge touches (a lone paper → phrase
  triple), and extracted edges whose object is a whole statement (more than 8 words).
- **Funnel** (``extraction_batches.json``): passages read, open propositions extracted, mapped onto
  the subschema, kept by evidence review; where the facts of a thin answer were lost.
- **Near duplicates**: entity/value nodes that are the same words up to plurals and word order
  ("attack task" / "attack tasks"), a sign that nodes were not merged.
- **Schema hygiene** (``subschema.ttl``): invented classes and properties, and how many invented
  properties are typed as OWL 2 object/datatype properties or have a parent property.
- **Export** (``triples.ttl``): its size and DBLP share next to the answer graph.

Nothing is enforced here; the numbers decide what Phase 4 should change.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from statistics import median
from typing import Any

from harness_schema.models import AnswerGraph, GraphEdge, GraphNode, Reading, Run
from harness_semantic import CORE_TERMS, DBLP, HRN, hand_written_subschema
from rdflib import OWL, RDF, RDFS, Graph

KNOWLEDGE_EDGES = {"fact", "claim", "derived"}
BACKBONE_PREFIX = "b-"   # edge ids of DBLP backbone facts (harness_baseline.backbone)
STATEMENT_WORDS = 8
STOP = {"a", "an", "the", "of", "for", "in", "on", "and", "to", "with", "by", "from"}


@dataclass(frozen=True)
class Vocabulary:
    dblp: frozenset[str]            # predicate names counted as DBLP (casefolded local names)
    seed: frozenset[str]            # other harness seed predicates
    seed_classes: frozenset[str]    # harness seed classes and DBLP classes


def _local(iri: object) -> str:
    return re.split(r"[#/]", str(iri))[-1]


@cache
def vocabulary() -> Vocabulary:
    schema = hand_written_subschema()
    graph = schema.graph
    dblp_props = {_local(p) for p in graph.subjects(RDF.type, RDF.Property) if str(p).startswith(str(DBLP))}
    equivalent = {_local(p) for p, _ in graph.subject_objects(OWL.equivalentProperty)
                  if str(p).startswith(str(HRN))}
    # The harness core, including the generic parents (researchRelation, ResearchEntity, ...).
    seed = ({_local(p) for p in schema.properties} | {_local(t) for t in CORE_TERMS}) - equivalent
    classes = {_local(c) for c in schema.classes} | {_local(t) for t in CORE_TERMS if _local(t)[:1].isupper()}
    classes |= {_local(c) for c in graph.subjects(RDF.type, OWL.Class) if str(c).startswith(str(DBLP))}
    return Vocabulary(
        dblp=frozenset(name.casefold() for name in dblp_props | equivalent | {"author"}),
        seed=frozenset(name.casefold() for name in seed),
        seed_classes=frozenset(name.casefold() for name in classes),
    )


def predicate_origin(label: str, vocab: Vocabulary | None = None) -> str:
    vocab = vocab or vocabulary()
    name = _local(label).casefold()
    if name in vocab.dblp:
        return "dblp"
    if name in vocab.seed:
        return "seed"
    return "invented"


def chosen_reading(run: Run) -> Reading | None:
    if run.chosen_reading is not None and 0 <= run.chosen_reading < len(run.readings):
        return run.readings[run.chosen_reading]
    return next((reading for reading in run.readings if reading.triples), None)


def _graph_of(reading: Reading) -> AnswerGraph:
    """The graph the UI shows; older runs without one get a plain graph from their triples."""
    if reading.graph is not None and reading.graph.nodes:
        return reading.graph
    nodes: dict[str, GraphNode] = {}
    edges = []
    for index, triple in enumerate(reading.triples):
        for value in (triple.s, triple.o):
            nodes.setdefault(value, GraphNode(id=value, label=value, kind="entity"))
        edges.append(GraphEdge(id=f"t{index}", source=triple.s, target=triple.o, label=triple.p, kind="fact"))
    return AnswerGraph(nodes=list(nodes.values()), edges=edges)


def _components(nodes: Iterable[str], edges: Iterable[tuple[str, str]]) -> list[set[str]]:
    parent: dict[str, str] = {node: node for node in nodes}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for source, target in edges:
        parent.setdefault(source, source)
        parent.setdefault(target, target)
        root_a, root_b = find(source), find(target)
        if root_a != root_b:
            parent[root_a] = root_b
    groups: dict[str, set[str]] = {}
    for node in parent:
        groups.setdefault(find(node), set()).add(node)
    return list(groups.values())


def _key(label: str) -> str:
    words = re.findall(r"[a-z0-9]+", label.casefold())
    folded = {word[:-1] if len(word) > 3 and word.endswith("s") else word for word in words} - STOP
    return " ".join(sorted(folded))


def _share(part: int, whole: int) -> float | None:
    return round(part / whole, 3) if whole else None


def _schema_hygiene(path: Path, vocab: Vocabulary) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        graph = Graph().parse(path, format="turtle")
    except Exception:  # noqa: BLE001 - a broken export is itself a finding
        return {"error": "subschema.ttl does not parse"}
    properties = {p for p in graph.subjects(RDF.type, RDF.Property) if str(p).startswith(str(HRN))}
    properties |= {p for kind in (OWL.ObjectProperty, OWL.DatatypeProperty)
                   for p in graph.subjects(RDF.type, kind) if str(p).startswith(str(HRN))}
    invented = [p for p in properties if predicate_origin(str(p), vocab) == "invented"]
    classes = {c for kind in (RDFS.Class, OWL.Class) for c in graph.subjects(RDF.type, kind)
               if str(c).startswith(str(HRN))}
    invented_classes = [c for c in classes if _local(c).casefold() not in vocab.seed_classes]

    def has(p: object, predicate: object) -> bool:
        return graph.value(p, predicate) is not None

    return {
        "invented_properties": sorted(_local(p) for p in invented),
        "invented_classes": sorted(_local(c) for c in invented_classes),
        "owl_typed": sum(any((p, RDF.type, kind) in graph for kind in (OWL.ObjectProperty, OWL.DatatypeProperty))
                         for p in invented),
        "with_parent": sum(has(p, RDFS.subPropertyOf) for p in invented),
        "with_domain_and_range": sum(has(p, RDFS.domain) and has(p, RDFS.range) for p in invented),
        "with_label_and_comment": sum(has(p, RDFS.label) and has(p, RDFS.comment) for p in invented),
    }


def _export(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        graph = Graph().parse(path, format="turtle")
    except Exception:  # noqa: BLE001
        return {"error": "triples.ttl does not parse"}
    predicates = Counter(str(p) for p in graph.predicates())
    dblp = sum(count for p, count in predicates.items() if p.startswith(str(DBLP)))
    return {"triples": len(graph), "dblp_share": _share(dblp, len(graph))}


def funnel(folder: Path) -> dict[str, Any] | None:
    """Where facts are lost: passages read, open propositions, mapped onto the schema, kept by review."""
    path = folder / "extraction_batches.json"
    if not path.exists():
        return None
    try:
        batches = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    passages = opened = mapped = 0
    kept: int | None = None
    for entry in batches if isinstance(batches, list) else []:
        if not isinstance(entry, dict):
            continue
        passages += int(entry.get("passages") or 0)
        for stage in entry.get("edc") or []:
            if stage.get("stage") == "extract":
                opened += int(stage.get("open_triples") or 0)
            elif stage.get("stage") == "canonicalize":
                mapped += sum(bool(item.get("selected")) for item in stage.get("decisions") or [])
        if entry.get("stage") == "evidence_review":
            kept = int(entry.get("kept") or 0)
    return {"passages": passages, "open": opened, "mapped": mapped, "kept": mapped if kept is None else kept}


def shape(run: Run, folder: Path | None = None) -> dict[str, Any] | None:
    """Graph-shape numbers for one run's chosen reading (None when it has no reading)."""
    reading = chosen_reading(run)
    if reading is None:
        return None
    vocab = vocabulary()
    graph = _graph_of(reading)
    kinds = {node.id: node.kind for node in graph.nodes}
    labels = {node.id: node.label for node in graph.nodes}
    edges = [edge for edge in graph.edges if edge.kind in KNOWLEDGE_EDGES]
    knowledge_nodes = {node.id for node in graph.nodes if node.kind != "passage"}
    degree: Counter[str] = Counter()
    for edge in edges:
        degree[edge.source] += 1
        degree[edge.target] += 1
    touched = {node for node in knowledge_nodes if degree[node]}
    # Literal values of facts (a year, a number, a phrase) do not connect papers, even when two
    # edges share one: they are attributes of their node. Claim context nodes (a benchmark, a metric)
    # are also "value" nodes, but they are meant to join results, so they do link.
    claim_touched = {node for edge in edges if edge.kind != "fact" for node in (edge.source, edge.target)}
    literals = {node for node in touched if kinds.get(node) == "value" and node not in claim_touched}
    linking = [edge for edge in edges if edge.target not in literals and edge.source not in literals]
    structural = touched - literals
    pieces = _components(structural, ((edge.source, edge.target) for edge in linking))
    papers = [node for node in touched if kinds.get(node) == "paper"]
    paper_pairs = len(papers) * (len(papers) - 1) // 2
    connected_pairs = sum(len([n for n in piece if kinds.get(n) == "paper"]) *
                          (len([n for n in piece if kinds.get(n) == "paper"]) - 1) // 2 for piece in pieces)

    facts = [edge for edge in edges if edge.kind == "fact"]

    def origin_of(edge: GraphEdge) -> str:
        # Runs since Phase 4 carry the origin on the edge; older ones are classified by label.
        if edge.origin is not None:
            return "seed" if edge.origin == "core" else edge.origin
        return predicate_origin(edge.label, vocab)

    origin = Counter(origin_of(edge) for edge in facts)
    invented = [edge for edge in facts if origin_of(edge) == "invented"]
    backbone = [edge for edge in facts if edge.id.startswith(BACKBONE_PREFIX)]
    # Shares of what the run extracted itself: the backbone would dilute them.
    extracted = [edge for edge in facts if not edge.id.startswith(BACKBONE_PREFIX)]
    dead_ends = [edge for edge in invented if degree[edge.target] <= 1]
    # DBLP values (a paper title) are long by nature; only extracted objects count as statements.
    statements = [edge for edge in extracted if origin_of(edge) != "dblp"
                  and len(labels.get(edge.target, "").split()) > STATEMENT_WORDS]
    literal_objects = [edge for edge in extracted if kinds.get(edge.target) == "value"]

    keys = Counter(_key(labels[node]) for node in touched if kinds.get(node) in {"entity", "value"} and _key(labels[node]))
    near_duplicates = sum(count for count in keys.values() if count > 1)

    inferred_props = [name for name in reading.properties if predicate_origin(name, vocab) == "invented"]
    inferred_classes = [name for name in reading.classes if name.casefold() not in vocab.seed_classes]
    root = folder / run.id if folder is not None else None
    # Metadata questions are answered from verified DBLP facts; research questions from passages.
    answered_from_facts = any(check.name == "verified facts" and check.status == "pass" for check in run.checks)
    return {
        "id": run.id, "path": run.path, "status": run.status, "question": run.question,
        "kind": "facts" if answered_from_facts else "research",
        "reading": reading.label,
        "nodes": len(touched), "edges": len(edges), "fact_edges": len(facts),
        "backbone_edges": len(backbone), "extracted_edges": len(extracted),
        "extracted_invented_share": _share(sum(origin_of(e) == "invented" for e in extracted), len(extracted)),
        "claim_edges": sum(edge.kind == "claim" for edge in edges),
        "dblp": origin["dblp"], "seed": origin["seed"], "invented": origin["invented"],
        "dblp_share": _share(origin["dblp"], len(facts)),
        "invented_share": _share(origin["invented"], len(facts)),
        "pieces": len(pieces),
        "largest_piece_share": _share(max((len(piece) for piece in pieces), default=0), len(structural)),
        "papers": len(papers),
        "paper_pairs_connected": _share(connected_pairs, paper_pairs),
        "invented_dead_end_share": _share(len(dead_ends), len(invented)),
        "literal_object_share": _share(len(literal_objects), len(extracted)),
        "statement_object_share": _share(len(statements), len(extracted)),
        "near_duplicate_nodes": near_duplicates,
        "invented_properties": sorted(set(inferred_props)),
        "invented_classes": sorted(set(inferred_classes)),
        "examples": {
            "dead_ends": [f"{labels.get(e.source, e.source)} —{e.label}→ {labels.get(e.target, e.target)}"
                          for e in dead_ends[:3]],
            "statements": [f"{labels.get(e.source, e.source)} —{e.label}→ {labels.get(e.target, e.target)}"
                           for e in statements[:2]],
        },
        "schema": _schema_hygiene(root / "subschema.ttl", vocab) if root else None,
        "export": _export(root / "triples.ttl") if root else None,
        "funnel": funnel(root) if root else None,
    }


SUMMARY_FIELDS = ("fact_edges", "backbone_edges", "extracted_edges", "dblp_share", "invented_share", "pieces", "largest_piece_share",
                  "paper_pairs_connected", "invented_dead_end_share", "literal_object_share",
                  "statement_object_share", "near_duplicate_nodes")


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Medians per kind (research / facts) and path, and per kind over all paths.

    The two kinds are kept apart: metadata answers are DBLP facts by construction, and mixing them
    in would hide how research answers look.
    """
    out = []
    groups = [(kind, path) for kind in ("research", "facts")
              for path in [*sorted({row["path"] for row in rows if row.get("kind") == kind}), "all"]]
    for kind, path in groups:
        group = [row for row in rows if row.get("kind") == kind and (path == "all" or row["path"] == path)]
        if not group:
            continue
        summary: dict[str, Any] = {"kind": kind, "path": path, "runs": len(group)}
        for name in SUMMARY_FIELDS:
            values = [row[name] for row in group if row.get(name) is not None]
            summary[name] = round(median(values), 3) if values else None
        invented = Counter(name for row in group for name in row["invented_properties"])
        summary["top_invented_properties"] = invented.most_common(8)
        funnels = [row["funnel"] for row in group if row.get("funnel")]
        summary["funnel"] = {key: round(median(item[key] for item in funnels), 1)
                             for key in ("passages", "open", "mapped", "kept")} if funnels else None
        summary["runs_with_invented_classes"] = sum(bool(row["invented_classes"]) for row in group)
        out.append(summary)
    return out


__all__ = ["Vocabulary", "chosen_reading", "predicate_origin", "shape", "summarize", "vocabulary"]
