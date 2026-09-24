"""The supervisor's tool loop against a scripted model and a fake MCP session."""

import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import ClassVar

import harness_agent.mcp as mcp_module
from harness_agent.mcp import MCPToolClient

SEARCH = {"total": 1, "items": [{"passage_id": "p4-intro", "paper_id": "p4", "text": "We introduce VPP."}]}


class FakeSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def initialize(self) -> None:
        pass

    async def list_tools(self):
        tool = SimpleNamespace(name="search_evidence", description="Search passages.",
                               inputSchema={"type": "object", "properties": {"query": {"type": "string"}}})
        return SimpleNamespace(tools=[tool])

    async def call_tool(self, name: str, arguments: dict):
        self.calls.append((name, arguments))
        return {"content": [{"type": "text", "text": json.dumps(SEARCH)}]}


class FakeResponse:
    status_code = 200

    def __init__(self, message: dict) -> None:
        self.message = message
        self.text = json.dumps(message)

    def json(self) -> dict:
        return {"choices": [{"message": self.message}]}


class ScriptedModelClient:
    """Plays one scripted assistant message per turn and records what it was sent."""

    script: ClassVar[list[dict]] = []
    sent: ClassVar[list[list[dict]]] = []

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url: str, *, timeout, headers, json: dict):
        ScriptedModelClient.sent.append(json["messages"])
        return FakeResponse(ScriptedModelClient.script.pop(0))


def tool_call(name: str, arguments: dict, call_id: str = "call-1") -> dict:
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}]}


def final(decision: dict) -> dict:
    return {"role": "assistant", "content": json.dumps(decision)}


def run_loop(monkeypatch, script: list[dict]) -> tuple[str, FakeSession, list[dict]]:
    session = FakeSession()

    @asynccontextmanager
    async def fake_transport(*args, **kwargs):
        yield (None, None, None)

    monkeypatch.setattr(mcp_module, "streamablehttp_client", fake_transport)
    monkeypatch.setattr(mcp_module, "ClientSession", lambda read, write: session)
    monkeypatch.setattr(mcp_module.httpx, "AsyncClient", ScriptedModelClient)
    monkeypatch.setenv("HARNESS_AGENT_CONTEXT_TOKENS", "32000")
    ScriptedModelClient.script = list(script)
    ScriptedModelClient.sent = []
    records: list[dict] = []
    model = SimpleNamespace(model="granite", base_url="http://model", api_key="k", timeout_s=30)
    client = MCPToolClient(records.append)
    answer = client.run_agent(model, "Which paper introduced VPP?", {
        "answer_source": "research",
        "preliminary_claims": [{"id": "c1", "kind": "MetadataClaim", "paper_id": "p4", "fields": {}, "source_ids": ["p4-metadata"]}],
    })
    return answer, session, records


def test_invented_evidence_is_sent_back_once_and_the_corrected_decision_is_accepted(monkeypatch) -> None:
    corrected = {"verified_claim_ids": ["c1"], "additional_evidence_ids": ["p4-intro"], "reasoning_summary": "fixed"}
    answer, session, records = run_loop(monkeypatch, [
        tool_call("search_evidence", {"query": "VPP"}),
        final({"verified_claim_ids": ["c1"], "additional_evidence_ids": ["p99-invented"], "reasoning_summary": "guess"}),
        final(corrected),
    ])
    assert json.loads(answer) == corrected
    assert len(session.calls) == 1                       # the corpus was searched once
    correction = ScriptedModelClient.sent[2][-1]
    assert correction["role"] == "user" and "p99-invented" in correction["content"]
    assert ScriptedModelClient.sent[2][-2]["content"] and "p99-invented" in ScriptedModelClient.sent[2][-2]["content"]
    assert [record["tool"] for record in records] == ["search_evidence"]


def test_reflection_budget_is_one_then_the_decision_is_returned_as_is(monkeypatch) -> None:
    bad = {"verified_claim_ids": [], "additional_evidence_ids": ["p99-invented"], "reasoning_summary": "still guessing"}
    answer, _session, _records = run_loop(monkeypatch, [tool_call("search_evidence", {"query": "VPP"}), final(bad), final(bad)])
    assert json.loads(answer) == bad                     # the caller's validation degrades it; no endless loop
    assert len(ScriptedModelClient.sent) == 3


def test_identical_tool_call_is_answered_from_the_ledger_without_hitting_the_server(monkeypatch) -> None:
    ok = {"verified_claim_ids": ["c1"], "additional_evidence_ids": ["p4-intro"], "reasoning_summary": "done"}
    _answer, session, records = run_loop(monkeypatch, [
        tool_call("search_evidence", {"query": "VPP"}),
        tool_call("search_evidence", {"query": "VPP"}, "call-2"),
        final(ok),
    ])
    assert len(session.calls) == 1
    assert [record["status"] for record in records] == ["ok", "repeat"]
    repeated = json.loads(ScriptedModelClient.sent[2][-1]["content"])
    assert repeated["repeated_call"] == records[0]["call_id"] and repeated["result"] == SEARCH
