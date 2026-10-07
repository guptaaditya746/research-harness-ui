"""The independent verifier finds the text of every cited id a tool returned in the session."""

import json
from types import SimpleNamespace

from harness_agent.verification import excerpt_index, independent_verification


class Verifier:
    model = "judge"

    def __init__(self) -> None:
        self.requests: list[dict] = []

    def complete_json(self, _system: str, user: str) -> dict:
        request = json.loads(user)
        self.requests.append(request)
        return {"supported_ids": sorted(request["cited_evidence"]), "unsupported_ids": [], "notes": ""}


AGGREGATE = {"tool": "aggregate_facts", "status": "ok", "result": {"total": 1, "items": [
    {"value": "VenueZ", "count": 2, "paper_ids": ["p1", "p2"], "fact_ids": ["p1-venue", "p2-venue"],
     "source_ids": ["p1-metadata", "p2-metadata"]}]}}
METADATA = {"tool": "metadata_table", "status": "ok", "result": {"items": [
    {"paper_id": "p3", "title": "MethodX for BenchY", "year": 2024}]}}
SEARCH = {"tool": "search_evidence", "status": "ok", "result": {"items": [
    {"passage_id": "p4-intro", "paper_id": "p4", "text": "We introduce MethodX."}]}}
READER = {"tool": "paper_reader", "status": "ok", "result": {
    "passage_ids": ["p5-2"], "passages": [{"passage_id": "p5-2", "text": "MethodX reaches 91% on BenchY."}],
    "findings": [{"statement": "a reader's claim, not evidence", "evidence_ids": ["p5-2", "p5-9"]}]}}
DOC_SEARCH = {"tool": "doc_search", "status": "ok", "result": {"items": [
    {"triple_id": "t00001", "subject": "p97", "predicate": "evaluatesOn", "object": "BenchY", "paper_id": "p97",
     "source_id": "p97-0", "evidence": "MethodX is evaluated on the BenchY benchmark."}]}}


def test_every_way_a_result_shows_an_id_yields_an_excerpt() -> None:
    wanted = {"p1-venue", "p2-metadata", "p3", "p4-intro", "p4", "p5-2", "p5-9", "t00001", "p97-0", "p77"}
    excerpts = excerpt_index([AGGREGATE, METADATA, SEARCH, READER, DOC_SEARCH], wanted)
    assert "VenueZ" in excerpts["p1-venue"] and "VenueZ" in excerpts["p2-metadata"]   # members of *_ids lists
    assert "MethodX for BenchY" in excerpts["p3"]                                     # a paper_id-only row
    assert excerpts["p4-intro"] == "We introduce MethodX."
    assert "We introduce MethodX." in excerpts["p4"]                                  # the passage's paper
    assert excerpts["p5-2"] == "MethodX reaches 91% on BenchY."                       # a reader's passage
    assert excerpts["t00001"] == "p97 evaluatesOn BenchY — passage: MethodX is evaluated on the BenchY benchmark."
    assert "BenchY benchmark" in excerpts["p97-0"]                                    # the triple's source passage
    # Not a tool result: a reader's own finding, and an id nothing returned.
    assert "p5-9" not in excerpts and "p77" not in excerpts


def test_only_ids_absent_from_every_result_are_unsupported_unjudged() -> None:
    verifier = Verifier()
    base_run = SimpleNamespace(readings=[SimpleNamespace(claims=[SimpleNamespace(
        id="c1", fields={}, evidence={"p8-metadata": "Paper p8 appeared at VenueZ."})])])
    cited = ["p1-venue", "p3", "p5-2", "t00001", "p8-metadata", "p77"]
    result, check = independent_verification(
        verifier, "q", {"reasoning_summary": "s"}, [], cited, [AGGREGATE, METADATA, READER, DOC_SEARCH], base_run)
    [request] = verifier.requests
    assert set(request["cited_evidence"]) == {"p1-venue", "p3", "p5-2", "t00001", "p8-metadata"}
    assert request["evidence_not_found_in_session"] == ["p77"]
    assert result["unsupported_ids"] == ["p77"] and check.status == "warn"
    assert sorted(result["supported_ids"]) == ["p1-venue", "p3", "p5-2", "p8-metadata", "t00001"]


def test_a_named_row_wins_over_a_listing_and_the_fuller_text_over_a_shorter_one() -> None:
    listing = {"tool": "aggregate_facts", "result": {"items": [{"value": "v", "fact_ids": ["f1"]}]}}
    fact = {"tool": "structured_facts", "result": {"items": [{"fact_id": "f1", "subject": "p1", "predicate": "year",
                                                              "object": "2024"}]}}
    short = {"tool": "doc_search", "result": {"items": [{"triple_id": "t1", "subject": "a", "predicate": "b",
                                                         "object": "c", "evidence": "short"}]}}
    full = {"tool": "doc_read", "result": {"items": [{"triple_id": "t1", "subject": "a", "predicate": "b",
                                                      "object": "c", "evidence": "short, and the rest"}]}}
    excerpts = excerpt_index([listing, fact, short, full], {"f1", "t1"})
    assert excerpts == {"f1": "p1 year 2024", "t1": "a b c — passage: short, and the rest"}
