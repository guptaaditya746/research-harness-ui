"""The answer as one connected graph, built with networkx.

Every entity appears once (three papers by the same author share one author
node). Source facts and claim fields are ``fact``/``claim``/``evidence`` edges
that keep their source ids; conclusions the harness computed (a count, an
extreme) are separate ``computed`` nodes joined by ``derived`` edges to the
facts they were computed from. Nothing is added that the answer did not use.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import networkx as nx
from harness_schema.models import (
    AnswerGraph,
    ClaimSubgraph,
    ComputedResult,
    GraphEdge,
    GraphNode,
    Triple,
)

VALUE_PREDICATES = {"year", "venue", "yearofpublication", "publishedin", "title"}
CONTEXT_FIELDS = ("value", "unit", "metric", "benchmark", "setting", "comparator", "changeType")
# The number itself belongs to its claim; the context it is measured in is shared, so two results
# on the same benchmark meet at one benchmark node while "10 %" in two papers never merges.
LOCAL_FIELDS = ("value", "unit", "changeType")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(text).casefold()).strip("-")[:80] or "x"


class _Builder:
    def __init__(self, is_paper: Callable[[str], bool], paper_label: Callable[[str], str]) -> None:
        self.graph = nx.MultiDiGraph()
        self.is_paper = is_paper
        self.paper_label = paper_label

    def node(self, value: str, *, role: str) -> str:
        """Add (or reuse) the node for ``value``; ``role`` hints its kind."""
        if role != "value" and self.is_paper(value):
            node_id, kind, label, iri = f"paper:{value}", "paper", self.paper_label(value), value
        elif role == "author":
            node_id, kind, label, iri = f"author:{_slug(value)}", "author", value, None
        elif role == "value":
            node_id, kind, label, iri = f"value:{_slug(value)}", "value", value, None
        else:
            node_id, kind, label, iri = f"entity:{_slug(value)}", "entity", value, None
        if not self.graph.has_node(node_id):
            self.graph.add_node(node_id, label=label, kind=kind, iri=iri)
        return node_id

    def add(self, source: str, target: str, *, label: str, kind: str, source_ids: list[str],
            evidence: str | None, edge_id: str, origin: str | None = None) -> str:
        self.graph.add_edge(source, target, key=edge_id, id=edge_id, label=label, kind=kind,
                            source_ids=source_ids, evidence=evidence, origin=origin)
        return edge_id


def build(triples: list[Triple], claims: list[ClaimSubgraph], computed: list[dict[str, Any]], *,
          is_paper: Callable[[str], bool], paper_label: Callable[[str], str],
          origin_of: Callable[[str], str | None] | None = None) -> AnswerGraph:
    """``origin_of`` names where a fact edge's predicate comes from (dblp / core / invented)."""
    builder = _Builder(is_paper, paper_label)
    edge_index: dict[tuple[str, str, str], str] = {}

    # Result/comparison claims with context become a claim node wired to each field and its passage,
    # instead of one flat triple, so a number is connected to its metric, benchmark and evidence.
    represented: set[tuple[str, str, str]] = set()
    for claim in claims:
        context = {name: field for name, field in claim.fields.items() if name in CONTEXT_FIELDS}
        if not context:
            continue
        subject = claim.fields.get("subject")
        obj = claim.fields.get("object")
        claim_id = f"claim:{claim.id}"
        local = [context[name].value for name in LOCAL_FIELDS if name in context]
        label = " ".join(local[:2]) + (f" ({local[2]})" if len(local) > 2 else "") if local else (
            obj.value if obj else claim.kind)
        builder.graph.add_node(claim_id, label=label, kind="claim", iri=None)
        paper = builder.node(claim.paper_id, role="subject")
        builder.add(paper, claim_id, label="reports", kind="claim", source_ids=claim.source_ids,
                    evidence=None, edge_id=f"e-{claim.id}-reports")
        for name, field in context.items():
            if name in LOCAL_FIELDS:
                continue
            target = builder.node(field.value, role="value")
            builder.add(claim_id, target, label=name, kind="claim", source_ids=field.source_ids,
                        evidence=field.quote, edge_id=f"e-{claim.id}-{name}")
        for source_id in claim.source_ids:
            passage = f"passage:{source_id}"
            builder.graph.add_node(passage, label=source_id, kind="passage", iri=None)
            builder.add(claim_id, passage, label="evidence", kind="evidence", source_ids=[source_id],
                        evidence=claim.evidence.get(source_id), edge_id=f"e-{claim.id}-ev-{_slug(source_id)}")
        if subject and obj and claim.fields.get("relation"):
            represented.add((subject.value, claim.fields["relation"].value, obj.value))

    for index, triple in enumerate(triples):
        if (triple.s, triple.p, triple.o) in represented:
            continue
        predicate = triple.p.casefold()
        role = "author" if predicate in {"authoredby", "author"} else (
            "value" if predicate in VALUE_PREDICATES else "object")
        source = builder.node(triple.s, role="subject")
        target = builder.node(triple.o, role=role)
        edge_id = triple.triple_id or f"t-{index:03d}"
        builder.add(source, target, label=triple.p, kind="fact", source_ids=[triple.source],
                    evidence=triple.evidence, edge_id=edge_id,
                    origin=origin_of(triple.p) if origin_of else None)
        edge_index[(triple.s, triple.p.casefold(), triple.o)] = edge_id

    results: list[ComputedResult] = []
    for number, item in enumerate(computed):
        about_role = item["about"]["role"]
        about = builder.node(item["about"]["value"], role=about_role)
        supporting = [
            edge_id for (s, p, o), edge_id in edge_index.items()
            if any(p == rule["p"].casefold() and (
                ("o" in rule and o == rule["o"]) or ("o_in" in rule and o in rule["o_in"]) or
                ("s_in" in rule and s in rule["s_in"]))
                for rule in item["support"])
        ]
        node_id = f"computed:{number}"
        builder.graph.add_node(node_id, label=item["label"], kind="computed", iri=None)
        builder.add(node_id, about, label="computed for", kind="derived", source_ids=[], evidence=item["operation"],
                    edge_id=f"d-{number}")
        results.append(ComputedResult(id=node_id, label=item["label"], operation=item["operation"],
                                      about=about, derivedFrom=supporting))

    graph = builder.graph
    return AnswerGraph(
        nodes=[GraphNode(id=node, label=str(data["label"]), kind=data["kind"], iri=data.get("iri"))
               for node, data in sorted(graph.nodes(data=True))],
        edges=[GraphEdge(id=data["id"], source=source, target=target, label=data["label"], kind=data["kind"],
                         sourceIds=list(data["source_ids"]), evidence=data["evidence"], origin=data.get("origin"))
               for source, target, data in sorted(graph.edges(data=True), key=lambda row: row[2]["id"])],
        computed=results,
    )
