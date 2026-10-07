"""The Agentic path with the evidence document D as input: D-only tools, D's readings, the agent's choice."""

import asyncio
import json
from pathlib import Path

from harness_agent import AgenticResearch
from harness_agent.doc_tools import DOC_TOOLS, EvidenceDocSession, chosen_triples
from harness_agent.mcp import MCPToolClient
from harness_baseline import Baseline20
from harness_baseline import evidence_doc as docs
from harness_corpus.store import CorpusStore
from harness_extraction import ExtractionEngine
from harness_schema.models import ConceptChoice
from harness_semantic import hand_written_subschema

QUESTION = "Which benchmarks does MethodX evaluate on?"


def setup(tmp_path: Path, readings: int = 1) -> tuple[CorpusStore, docs.EvidenceDoc]:
    root = tmp_path / "corpus"
    root.mkdir()
    texts = ["MethodX is evaluated on the BenchY benchmark.", "MethodX is trained with the AdamW optimizer.",
             "MethodX is also evaluated on BenchW."]
    rows = {"papers.jsonl": [{"paper_id": "p97", "title": "MethodX", "year": 2025}],
            "passages.jsonl": [{"passage_id": f"p97-{n}", "paper_id": "p97", "section_path": "Body", "para_idx": n,
                                "text": text} for n, text in enumerate(texts)],
            "facts.jsonl": [], "references.jsonl": []}
    for name, values in rows.items():
        (root / name).write_text("".join(json.dumps(row) + "\n" for row in values))
    store = CorpusStore(root)
    triples = [{"id": f"t0000{n + 1}", "s": "p97", "p": p, "o": o, "paper_id": "p97",
                "sources": [{"source_id": f"p97-{n}", "evidence": texts[n]}]}
               for n, (p, o) in enumerate([("evaluatesOn", "BenchY"), ("usesOptimizer", "AdamW"),
                                           ("evaluatesOn", "BenchW")])]
    folder = tmp_path / "docs" / "q90"
    folder.mkdir(parents=True)
    (folder / "doc.json").write_text(json.dumps({"question": {"id": "q90", "question": QUESTION}, "interpretations": [
        {"id": f"i{n}", "label": f"Reading {n}", "gloss": f"gloss {n}", "triples": triples[n - 1:]}
        for n in range(1, readings + 1)]}))
    return store, docs.load(tmp_path / "docs", QUESTION, store.by_passage)


class PickingModel:
    model = "stub"

    def complete_json(self, system: str, user: str) -> dict:
        payload = json.loads(user)
        if "keep_ids" in system:
            return {"keep_ids": [c["id"] for c in payload["candidates"]], "drop_reasons": {}}
        sources = sorted({t["source_id"] for t in payload["triples"]})
        if "verdict" in system or "sentences" in json.dumps(payload):
            return {"sentences": [{"text": "x", "verdict": "supported", "sourceIds": sources}]}
        return {"sparql": "SELECT ?s ?o WHERE { ?s ?p ?o . }", "cited_source_ids": sources,
                "answer": "MethodX is evaluated on " + ", ".join(f"[{s}]" for s in sources) + "."}


def test_doc_tools_search_page_and_read_only_the_document(tmp_path: Path) -> None:
    _, doc = setup(tmp_path)
    session = EvidenceDocSession(doc)
    names = [tool.name for tool in asyncio.run(session.list_tools()).tools]
    assert tuple(names) == DOC_TOOLS
    overview = asyncio.run(session.call_tool("doc_overview", {}))
    assert overview["triples"] == 3 and dict(overview["relations"])["evaluatesOn"] == 2
    found = asyncio.run(session.call_tool("doc_search", {"query": "BenchY benchmark"}))
    assert found["items"][0]["triple_id"] == "t00001"
    page = asyncio.run(session.call_tool("doc_triples", {"predicate": "evaluatesOn", "limit": 1}))
    assert page["total"] == 2 and page["next_offset"] == 1
    read = asyncio.run(session.call_tool("doc_read", {"triple_ids": ["t00002", "t99"]}))
    assert read["items"][0]["evidence"].startswith("MethodX is trained") and read["unknown"] == ["t99"]
    assert "error" in asyncio.run(session.call_tool("search_evidence", {"query": "x"}))
    assert chosen_triples(doc, {"t00001", "p97-2"}) == {"t00001", "t00003"}


def test_doc_search_and_doc_triples_page_with_offset_and_say_where_they_stand(tmp_path: Path) -> None:
    _, doc = setup(tmp_path)
    session = EvidenceDocSession(doc)
    first = asyncio.run(session.call_tool("doc_search", {"query": "MethodX evaluated", "limit": 2}))
    assert first["total"] == 3 and first["returned"] == 2 and first["next_offset"] == 2
    assert first["showing"] == "1–2 of 3"
    rest = asyncio.run(session.call_tool("doc_search", {"query": "MethodX evaluated", "offset": first["next_offset"]}))
    assert rest["showing"] == "3–3 of 3" and "next_offset" not in rest
    seen = [item["triple_id"] for item in first["items"] + rest["items"]]
    assert sorted(seen) == ["t00001", "t00002", "t00003"]
    # A limit above four pages is cut to four pages, and the result says so instead of silently.
    capped = asyncio.run(session.call_tool("doc_triples", {"limit": 10_000}))
    assert capped["limit_capped_to"] == session.page * 4 and capped["showing"] == "1–3 of 3"
    beyond = asyncio.run(session.call_tool("doc_triples", {"offset": 9}))
    assert beyond["items"] == [] and beyond["showing"] == "none of 3 at offset 9"


def test_readings_are_the_documents_interpretations(tmp_path: Path) -> None:
    store, doc = setup(tmp_path, readings=2)
    baseline = Baseline20(store, ExtractionEngine(PickingModel(), hand_written_subschema()), refine=True)
    proposal = AgenticResearch(baseline, evidence_doc=doc).propose(QUESTION, "slow", "r-p")["run"]
    assert proposal.status == "awaiting_reading" and [c.label for c in proposal.concepts] == ["Reading 1", "Reading 2"]
    assert proposal.run_input == "evidence_doc"


def test_the_agent_chooses_triples_and_the_pipeline_answers_from_them(tmp_path: Path, monkeypatch) -> None:
    store, doc = setup(tmp_path)
    seen: dict = {}

    def run_agent(self, model, question, grounded=None, session=None):
        seen["grounded"], seen["session"] = grounded, session
        result = asyncio.run(session.call_tool("doc_triples", {"predicate": "evaluatesOn"}))
        self.sink({"call_id": "tool-001", "agent": "supervisor", "tool": "doc_triples", "arguments": {},
                   "result": result, "result_count": len(result["items"]), "duration_ms": 1.0, "status": "ok"})
        return json.dumps({"reasoning_summary": "the evaluatesOn triples", "verified_claim_ids": [],
                           "additional_evidence_ids": ["t00001", "t00003", "t42"], "findings": []})

    monkeypatch.setattr(MCPToolClient, "run_agent", run_agent)
    baseline = Baseline20(store, ExtractionEngine(PickingModel(), hand_written_subschema()), refine=True)
    concept = ConceptChoice(label="Reading 1", gloss="gloss 1", classes=["Paper"])
    run = AgenticResearch(baseline, evidence_doc=doc).run(QUESTION, "slow", "r-a", concept=concept)["run"]
    assert isinstance(seen["session"], EvidenceDocSession) and seen["grounded"]["answer_source"] == "evidence_document"
    assert run.path == "agentic" and run.status == "answered", run.reason
    assert run.run_input == "evidence_doc" and run.evidence_doc["chosen_by_agent"] == 2
    assert sorted(t.o for t in run.readings[0].triples) == ["BenchW", "BenchY"]      # not the AdamW triple
    assert {c.name: c.status for c in run.checks}["agent choice"] == "pass"
    assert baseline.evidence_doc is None                                            # restored after the run
