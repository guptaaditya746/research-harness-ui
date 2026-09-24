from harness_baseline import answer_graph
from harness_schema.models import ClaimSubgraph, Triple


def _field(value: str, source: str) -> dict:
    return {"value": value, "sourceIds": [source], "quote": f"... {value} ..."}


def _result_claim(claim_id: str, paper: str, value: str, source: str) -> ClaimSubgraph:
    return ClaimSubgraph.model_validate({
        "id": claim_id, "kind": "ResultClaim", "paperId": paper, "sourceIds": [source],
        "evidence": {source: f"we reach {value} on CALVIN"},
        "fields": {"subject": _field(paper, source), "relation": _field("reportsResult", source),
                   "object": _field(f"{value} success", source), "value": _field(value, source),
                   "unit": _field("%", source), "metric": _field("success rate", source),
                   "benchmark": _field("CALVIN ABC→D", source)},
    })


def test_result_numbers_connect_to_metric_benchmark_paper_and_passage() -> None:
    claims = [_result_claim("c1", "p4", "41.5", "p4-results"), _result_claim("c2", "p6", "10", "p6-results")]
    triples = [Triple(s="p4", p="reportsResult", o="41.5 success", support="claim", evidence="x", source="p4-results"),
               Triple(s="p4", p="year", o="2025", support="fact", evidence="2025", source="p4-metadata")]
    graph = answer_graph.build(triples, claims, [], is_paper=lambda v: v in {"p4", "p6"}, paper_label=str.upper)
    nodes = {node.id: node for node in graph.nodes}
    assert nodes["claim:c1"].label == "41.5 %" and nodes["claim:c2"].label == "10 %"
    benchmark = [node for node in graph.nodes if node.label == "CALVIN ABC→D"]
    assert len(benchmark) == 1                                   # both results meet at one benchmark node
    into_benchmark = [edge for edge in graph.edges if edge.target == benchmark[0].id]
    assert {edge.source for edge in into_benchmark} == {"claim:c1", "claim:c2"}
    assert not any(node.label == "%" for node in graph.nodes)    # units never become shared hubs
    evidence = [edge for edge in graph.edges if edge.kind == "evidence" and edge.source == "claim:c1"]
    assert evidence[0].target == "passage:p4-results" and "41.5" in evidence[0].evidence
    # The claim replaces its flat triple; unrelated facts stay as fact edges on the same paper node.
    assert not any(edge.label == "reportsResult" for edge in graph.edges)
    assert any(edge.label == "year" and edge.source == "paper:p4" for edge in graph.edges)
    assert nodes["paper:p4"].label == "P4"
