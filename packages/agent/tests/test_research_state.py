import json

from harness_agent.mcp import decision_problems
from harness_agent.research_state import LOOP_HINT_AFTER, ResearchLedger


def _facts(paper: str) -> dict:
    return {"total": 1, "items": [{"fact_id": f"{paper}-y", "subject": paper, "predicate": "year", "object": "2025",
                                   "source_id": f"{paper}-metadata"}]}


def test_ledger_digest_keeps_what_each_call_found_after_compaction() -> None:
    ledger = ResearchLedger(question="q")
    ledger.record("search_evidence", {"total": 2, "items": [
        {"passage_id": "p4-abstract", "paper_id": "p4", "text": "We propose the Video Prediction Policy for robot learning."},
        {"passage_id": "p6-intro", "paper_id": "p6", "text": "FOCUS learns object-centric world models."},
    ]}, "collect", arguments={"query": "video prediction", "limit": 5}, call_id="tool-001")
    summary = json.loads(ledger.summary())
    [digest] = summary["found"]
    assert digest["tool"] == "search_evidence" and digest["args"] == {"query": "video prediction", "limit": 5}
    assert digest["items"][0].startswith("p4-abstract: p4 · We propose the Video Prediction Policy")
    assert {"p4-abstract", "p6-intro", "p4", "p6"} <= set(summary["evidence_ids"])


def test_identical_calls_are_answered_from_the_ledger() -> None:
    ledger = ResearchLedger(question="q")
    assert ledger.repeat_of("structured_facts", {"paper_id": "p4"}) is None
    ledger.record("structured_facts", _facts("p4"), "collect", arguments={"paper_id": "p4"}, call_id="tool-001")
    assert ledger.repeat_of("structured_facts", {"paper_id": "p4"}) == ("tool-001", _facts("p4"))
    assert ledger.repeat_of("structured_facts", {"paper_id": "p5"}) is None
    assert ledger.repeats == 1


def test_per_paper_loop_gets_one_aggregate_hint() -> None:
    ledger = ResearchLedger(question="q")
    hints = [ledger.loop_hint("structured_facts", {"paper_id": f"p{n}"}) for n in range(LOOP_HINT_AFTER + 3)]
    given = [hint for hint in hints if hint]
    assert len(given) == 1 and "aggregate_facts" in given[0] and hints[LOOP_HINT_AFTER - 1] is not None
    assert ledger.loop_hint("search_evidence", {"query": "x"}) is None


def test_decision_problems_name_exactly_what_to_fix() -> None:
    observed, claims = {"p4-abstract"}, {"c1"}
    ok = json.dumps({"verified_claim_ids": ["c1"], "additional_evidence_ids": ["p4-abstract"], "reasoning_summary": "x"})
    assert decision_problems(ok, observed=observed, claim_ids=claims, corpus_calls=1, needs_tools=True) == []
    invented = json.dumps({"verified_claim_ids": ["c9"], "additional_evidence_ids": ["p99-made-up"], "reasoning_summary": "x"})
    problems = decision_problems(invented, observed=observed, claim_ids=claims, corpus_calls=0, needs_tools=True)
    assert any("p99-made-up" in item for item in problems)
    assert any("c9" in item and "c1" in item for item in problems)
    assert any("no corpus tool" in item for item in problems)
    assert decision_problems("I think the answer is p4.", observed=observed, claim_ids=claims, corpus_calls=1, needs_tools=False) \
        == ["the final message must be one JSON object with keys verified_claim_ids, additional_evidence_ids, reasoning_summary"]
    # Answers computed from verified facts do not require a tool call to be accepted.
    assert decision_problems(ok, observed=observed, claim_ids=claims, corpus_calls=0, needs_tools=False) == []


def test_digest_names_triples_with_their_source_passage() -> None:
    ledger = ResearchLedger(question="q")
    ledger.record("doc_search", {"total": 1, "items": [
        {"subject": "p97", "predicate": "evaluatesOn", "object": "BenchY", "paper_id": "p97",
         "source_id": "p97-0", "evidence": "MethodX is evaluated on BenchY.", "triple_id": "t00001",
         "interpretation": "i1"},
    ]}, "collect", arguments={"query": "BenchY"}, call_id="tool-001")
    [line] = ledger.digests[0]["items"]
    assert line == "t00001 (source p97-0): p97 evaluatesOn BenchY"


def test_digest_keeps_the_first_ids_of_aggregate_lists() -> None:
    ledger = ResearchLedger(question="q")
    fact_ids = [f"p{n}-venue" for n in range(12)]
    ledger.record("aggregate_facts", {"total": 1, "items": [
        {"value": "VenueZ", "count": 12, "paper_ids": [f"p{n}" for n in range(12)], "fact_ids": fact_ids,
         "source_ids": ["p0-metadata", "p1-metadata"]},
    ]}, "collect", arguments={"operation": "count"}, call_id="tool-001")
    [line] = ledger.digests[0]["items"]
    assert line.startswith("VenueZ · 12 | paper_ids: p0, p1, p2, p3, p4 +7 more")
    assert "fact_ids: p0-venue, p1-venue, p2-venue, p3-venue, p4-venue +7 more" in line
    assert "source_ids: p0-metadata, p1-metadata" in line and "more |" not in line.split("source_ids")[1]
    huge = {"value": "x" * 500, "fact_ids": ["f" * 80 for _ in range(10)], "source_ids": ["s" * 80] * 10}
    ledger.record("aggregate_facts", {"items": [huge]}, "collect", arguments={"operation": "x"})
    [bounded] = ledger.digests[1]["items"]
    assert len(bounded) <= 320 and bounded.startswith("x" * 120 + " | fact_ids: ")
