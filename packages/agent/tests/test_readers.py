"""Parallel paper readers: which papers, what they may cite, and that they really run side by side."""

from __future__ import annotations

import json
import threading

from harness_agent import AgenticResearch
from harness_agent.mcp import MCPToolClient
from harness_agent.readers import choose_papers, read_paper, run_readers


class Corpus:
    def __init__(self) -> None:
        self.by_passage = {
            pid: {"passage_id": pid, "paper_id": pid.split("-")[0], "text": f"text of {pid}"}
            for pid in ("p1-a", "p1-b", "p2-a", "p2-b", "p3-a")
        }

    def hybrid_rank(self, query, scope=None, passage_ids=None, offset=0, limit=10):
        papers = (scope or {}).get("paper_ids")
        rows = [row for row in self.by_passage.values() if not papers or row["paper_id"] in papers]
        return {"items": rows[:limit], "total": len(rows)}

    def paper_title(self, paper_id: str) -> str:
        return paper_id.upper()


class ReaderModel:
    model = "lite-model"

    def __init__(self, reply=None, barrier: threading.Barrier | None = None) -> None:
        self.reply = reply
        self.barrier = barrier
        self.requests: list[dict] = []

    def complete_json(self, system: str, user: str) -> dict:
        payload = json.loads(user)
        self.requests.append(payload)
        if self.barrier is not None:
            self.barrier.wait(timeout=5)          # every reader must be in flight at once
        if self.reply is not None:
            return self.reply
        first = payload["passages"][0]["passage_id"]
        return {"findings": [{"statement": f"{payload['paper']['paper_id']} says something.",
                              "evidence_ids": [first, "p9-invented"]}]}


def test_papers_are_ranked_by_evidence_and_fall_back_to_the_question() -> None:
    corpus = Corpus()
    assert choose_papers(corpus, "q", [("p2-a", 3), ("p1-a", 1), ("p1-b", 1), ("x", 9)], 2) == ["p2", "p1"]
    assert choose_papers(corpus, "q", [], 2) == ["p1", "p2"]


def test_a_reader_cites_only_the_passages_it_was_shown_and_never_fails_the_run() -> None:
    result = read_paper(ReaderModel(), Corpus(), "What?", {"label": "L", "gloss": "G"}, "p2", 2)
    assert result.passage_ids == ["p2-a", "p2-b"]
    assert result.findings == [{"statement": "p2 says something.", "evidence_ids": ["p2-a"]}]

    class Broken(ReaderModel):
        def complete_json(self, system, user):
            raise TimeoutError("slow")

    failed = read_paper(Broken(), Corpus(), "What?", None, "p1", 2)
    assert failed.findings == [] and failed.error.startswith("TimeoutError")


def test_readers_run_in_parallel() -> None:
    model = ReaderModel(barrier=threading.Barrier(3))
    results = run_readers(model, Corpus(), "What?", None, ["p1", "p2", "p3"], 2)
    assert [result.paper_id for result in results] == ["p1", "p2", "p3"]
    assert all(result.error is None for result in results)


def test_reader_findings_join_the_hand_off(monkeypatch) -> None:
    monkeypatch.setattr(AgenticResearch, "_heavy_queue_depth", staticmethod(lambda: 0))
    monkeypatch.setenv("HARNESS_SUBAGENT_READERS", "2")
    from test_agentic import CONCEPT, StubBaseline

    class Baseline(StubBaseline):
        def __init__(self):
            self.seed_passage_ids = []
            self.agent_findings = []
            self.corpus = Corpus()

        def run(self, question, budget, run_id, concept=None):
            self.seen = (list(self.seed_passage_ids), list(self.agent_findings))
            return super().run(question, budget, run_id, concept=concept)

    def run_agent(self, model, question, grounded_context):
        self.sink({"call_id": "tool-001", "agent": "supervisor", "tool": "search_evidence",
                   "arguments": {"query": "q"}, "status": "ok", "duration_ms": 1.0, "result_count": 1,
                   "result": {"items": [{"passage_id": "p2-b"}]}})
        return json.dumps({"verified_claim_ids": [], "additional_evidence_ids": ["p2-b"], "reasoning_summary": "x"})

    monkeypatch.setattr(MCPToolClient, "run_agent", run_agent)
    baseline = Baseline()
    result = AgenticResearch(baseline, agent_model=ReaderModel()).run("Compare.", "slow", "r-read", concept=CONCEPT)
    seeds, findings = baseline.seen
    assert [item["source"] for item in findings] == ["reader:p2", "reader:p1"]
    # The supervisor's decision id first, then one reader passage from the other paper.
    assert seeds[:2] == ["p2-b", "p1-a"] and "p2-a" in seeds
    calls = [json.loads(line) for line in result["artifacts"]["tool_calls.jsonl"].splitlines() if line.strip()]
    assert [call["agent"] for call in calls if call["tool"] == "paper_reader"] == ["reader:p2", "reader:p1"]
    checks = {check.name: check for check in result["run"].checks}
    assert checks["paper readers"].status == "pass"
