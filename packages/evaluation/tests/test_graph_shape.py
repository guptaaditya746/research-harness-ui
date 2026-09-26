"""Graph-shape diagnostics on a small hand-made answer graph."""

from __future__ import annotations

from harness_evaluation.graph_shape import predicate_origin, shape, summarize
from harness_schema.models import AnswerGraph, GraphEdge, GraphNode, Metrics, Reading, Run


def node(node_id: str, kind: str, label: str | None = None) -> GraphNode:
    return GraphNode(id=node_id, label=label or node_id, kind=kind)


def edge(source: str, target: str, label: str, kind: str = "fact") -> GraphEdge:
    return GraphEdge(id=f"{source}-{label}-{target}", source=source, target=target, label=label, kind=kind)


def run_with(graph: AnswerGraph, properties: list[str], classes: list[str]) -> Run:
    reading = Reading(label="r", gloss="g", classes=classes, axioms=[], sparql="", answer="a", triples=[],
                      metrics=Metrics(), status="answered", graph=graph, properties=properties)
    return Run(id="r-1", startedAt="2026-01-01T00:00:00Z", durationS=1, question="q", path="agentic",
               budget="slow", status="answered", readings=[reading], chosenReading=0)


def test_origin_of_predicates() -> None:
    assert predicate_origin("authoredBy") == "dblp"                 # harness term equivalent to dblp:authoredBy
    assert predicate_origin("https://dblp.org/rdf/schema#publishedInStream") == "dblp"
    assert predicate_origin("usesDataset") == "seed"
    assert predicate_origin("evaluatesAttack") == "invented"


def test_shape_counts_mix_connectivity_dead_ends_and_duplicates() -> None:
    graph = AnswerGraph(
        nodes=[node("paper:a", "paper"), node("paper:b", "paper"), node("paper:c", "paper"),
               node("author:x", "author"), node("entity:t1", "entity", "attack task"),
               node("entity:t2", "entity", "attack tasks"),
               node("value:s", "value", "a long statement that the model copied as the object of one triple"),
               node("passage:p1", "passage")],
        edges=[edge("paper:a", "author:x", "authoredBy"), edge("paper:b", "author:x", "authoredBy"),
               edge("paper:a", "entity:t1", "evaluatesAttack"), edge("paper:b", "entity:t2", "evaluatesAttack"),
               edge("paper:c", "value:s", "findsLimitation"),
               edge("paper:a", "passage:p1", "evidence", kind="evidence")],
    )
    row = shape(run_with(graph, ["authoredBy", "evaluatesAttack", "findsLimitation"], ["Paper", "Attack"]))
    assert (row["dblp"], row["seed"], row["invented"]) == (2, 0, 3)
    assert row["dblp_share"] == 0.4 and row["invented_share"] == 0.6
    assert row["pieces"] == 2                                  # a–x–b (with their attacks) and c–statement
    assert row["paper_pairs_connected"] == round(1 / 3, 3)     # only a–b of the three pairs
    assert row["invented_dead_end_share"] == 1.0               # every invented object is touched once
    assert row["statement_object_share"] == 0.2
    assert row["near_duplicate_nodes"] == 2                    # "attack task" and "attack tasks"
    assert row["invented_properties"] == ["evaluatesAttack", "findsLimitation"]
    assert row["invented_classes"] == ["Attack"]
    assert row["kind"] == "research"
    [agentic, everything] = summarize([row])
    assert (agentic["kind"], agentic["path"]) == ("research", "agentic") and everything["runs"] == 1


def test_older_runs_without_a_graph_use_their_triples() -> None:
    from harness_schema.models import Triple

    reading = Reading(label="r", gloss="g", classes=[], axioms=[], sparql="", answer="a", metrics=Metrics(),
                      status="answered", triples=[Triple(s="p1", p="title", o="T", support="s", evidence="e",
                                                         source="x")])
    run = Run(id="r-2", startedAt="2026-01-01T00:00:00Z", durationS=1, question="q", path="baseline",
              budget="fast", status="answered", readings=[reading], chosenReading=0)
    assert shape(run)["dblp_share"] == 1.0


def test_answers_from_verified_facts_are_their_own_kind() -> None:
    from harness_schema.models import Check

    run = run_with(AnswerGraph(nodes=[node("paper:a", "paper"), node("author:x", "author")],
                               edges=[edge("paper:a", "author:x", "authoredBy")]), [], [])
    run = run.model_copy(update={"checks": [Check(name="verified facts", status="pass")]})
    row = shape(run)
    assert row["kind"] == "facts"
    assert [(item["kind"], item["path"]) for item in summarize([row])] == [("facts", "agentic"), ("facts", "all")]
