"""The supervisor's tool loop against a scripted model and a fake MCP session."""

import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import ClassVar

import harness_agent.mcp as mcp_module
import harness_agent.tool_loop as tool_loop_module
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
    bodies: ClassVar[list[dict]] = []

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url: str, *, timeout, headers, json: dict):
        ScriptedModelClient.sent.append(json["messages"])
        ScriptedModelClient.bodies.append(json)
        return FakeResponse(ScriptedModelClient.script.pop(0))


def tool_call(name: str, arguments: dict, call_id: str = "call-1") -> dict:
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}]}


def final(decision: dict) -> dict:
    return {"role": "assistant", "content": json.dumps(decision)}


def run_loop(monkeypatch, script: list[dict], window: str = "32000") -> tuple[str, FakeSession, list[dict]]:
    session = FakeSession()

    @asynccontextmanager
    async def fake_transport(*args, **kwargs):
        yield (None, None, None)

    monkeypatch.setattr(mcp_module, "streamablehttp_client", fake_transport)
    monkeypatch.setattr(mcp_module, "ClientSession", lambda read, write: session)
    monkeypatch.setattr(tool_loop_module.httpx, "AsyncClient", ScriptedModelClient)
    monkeypatch.setenv("HARNESS_AGENT_CONTEXT_TOKENS", window)
    ScriptedModelClient.script = list(script)
    ScriptedModelClient.sent = []
    records: list[dict] = []
    model = SimpleNamespace(model="granite", base_url="http://model", api_key="k", timeout_s=30)
    client = MCPToolClient(records.append)
    run_loop.client = client
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



def test_first_turn_requires_a_tool_call_then_the_agent_chooses(monkeypatch) -> None:
    ScriptedModelClient.bodies = []
    run_loop(monkeypatch, [
        tool_call("search_evidence", {"query": "VPP"}),
        final({"verified_claim_ids": [], "additional_evidence_ids": ["p4-intro"], "reasoning_summary": "ok"}),
    ])
    assert [body["tool_choice"] for body in ScriptedModelClient.bodies] == ["required", "auto"]


def test_a_stop_inside_the_mcp_session_reaches_the_runtime() -> None:
    from harness_agent.mcp import _first_non_exception

    class Cancelled(BaseException):
        pass

    stop = Cancelled("stopped")
    group = BaseExceptionGroup("task group", [ValueError("x"), BaseExceptionGroup("inner", [stop])])
    assert _first_non_exception(group) is stop
    assert _first_non_exception(ExceptionGroup("only errors", [ValueError("x")])) is None


def test_findings_are_kept_by_the_client_and_invented_ids_removed(monkeypatch) -> None:
    decision = {"verified_claim_ids": [], "additional_evidence_ids": ["p4-intro"], "reasoning_summary": "done",
                "findings": [{"statement": "VPP is introduced in p4.", "evidence_ids": ["p4-intro"]}]}
    answer, session, records = run_loop(monkeypatch, [
        tool_call("search_evidence", {"query": "VPP"}),
        tool_call("record_finding", {"statement": "VPP is introduced in p4.",
                                     "evidence_ids": ["p4-intro", "p99-invented"]}, "call-2"),
        tool_call("record_finding", {"statement": "Nothing to cite.", "evidence_ids": ["p77"]}, "call-3"),
        final(decision),
    ])
    assert json.loads(answer) == decision
    assert [name for name, _ in session.calls] == ["search_evidence"]      # answered locally, not by MCP
    assert run_loop.client.notes == [{"id": "F1", "statement": "VPP is introduced in p4.",
                                      "evidence_ids": ["p4-intro"]}]
    finding_records = [record for record in records if record["tool"] == "record_finding"]
    assert [record["status"] for record in finding_records] == ["ok", "error"]
    assert finding_records[0]["result"]["removed"] == ["p99-invented"]
    assert "record_finding" in [tool["function"]["name"] for tool in ScriptedModelClient.bodies[-1]["tools"]]


def test_findings_citing_unseen_ids_are_sent_back(monkeypatch) -> None:
    from harness_agent.mcp import decision_problems

    problems = decision_problems(
        json.dumps({"verified_claim_ids": [], "additional_evidence_ids": [], "reasoning_summary": "x",
                    "findings": [{"statement": "s", "evidence_ids": ["p99"]}]}),
        observed={"p4-intro"}, claim_ids=set(), corpus_calls=1, needs_tools=True)
    assert problems == ["finding 1 cites no id returned by a tool in this session"]


def _session_with(monkeypatch, names: list[str], text: str = "") -> None:
    async def list_tools(self):
        return SimpleNamespace(tools=[SimpleNamespace(name=name, description=name, inputSchema={"type": "object"})
                                      for name in names])

    monkeypatch.setattr(FakeSession, "list_tools", list_tools)
    if text:
        async def call_tool(self, name, arguments):
            self.calls.append((name, arguments))
            return {"content": [{"type": "text", "text": json.dumps(
                {"total": 1, "items": [{"passage_id": f"p{len(self.calls)}-a", "text": text}]})}]}

        monkeypatch.setattr(FakeSession, "call_tool", call_tool)


def test_curated_toolset_hides_duplicate_tools(monkeypatch) -> None:
    _session_with(monkeypatch, ["search_evidence", "rank", "query2doc", "read_evidence", "facts"])
    ScriptedModelClient.bodies = []
    run_loop(monkeypatch, [tool_call("search_evidence", {"query": "VPP"}),
                           final({"verified_claim_ids": [], "additional_evidence_ids": ["p4-intro"],
                                  "reasoning_summary": "ok"})])
    first, second = ([tool["function"]["name"] for tool in body["tools"]] for body in ScriptedModelClient.bodies)
    # The forced first turn ("required") offers corpus tools only; the scratchpad comes after.
    assert ScriptedModelClient.bodies[0]["tool_choice"] == "required"
    assert first == ["search_evidence", "read_evidence"]
    assert second == ["search_evidence", "read_evidence", "record_finding"]


def test_tool_budget_leaves_only_the_scratchpad_and_asks_for_the_decision(monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_AGENT_MAX_TOOL_CALLS", "1")
    ScriptedModelClient.bodies = []
    run_loop(monkeypatch, [tool_call("search_evidence", {"query": "VPP"}),
                           final({"verified_claim_ids": [], "additional_evidence_ids": ["p4-intro"],
                                  "reasoning_summary": "ok"})])
    second = ScriptedModelClient.bodies[1]
    assert [tool["function"]["name"] for tool in second["tools"]] == ["record_finding"]
    assert "tool budget" in second["messages"][-1]["content"]


class SummaryModel(SimpleNamespace):
    def complete_json(self, system: str, user: str) -> dict:
        self.summary_requests = [*getattr(self, "summary_requests", []), json.loads(user)]
        return {"summary": "searched three times; p1-a and p2-a matter", "open": ["benchmark numbers"]}


def test_near_the_limit_older_exchanges_become_a_summary(monkeypatch) -> None:
    _session_with(monkeypatch, ["search_evidence"], text="long passage " * 400)
    monkeypatch.setenv("HARNESS_AGENT_CONTEXT_TOKENS", "6000")
    monkeypatch.setenv("HARNESS_AGENT_KEEP_RESULTS", "0")
    session = FakeSession()

    @asynccontextmanager
    async def fake_transport(*args, **kwargs):
        yield (None, None, None)

    monkeypatch.setattr(mcp_module, "streamablehttp_client", fake_transport)
    monkeypatch.setattr(mcp_module, "ClientSession", lambda read, write: session)
    monkeypatch.setattr(tool_loop_module.httpx, "AsyncClient", ScriptedModelClient)
    ScriptedModelClient.script = [tool_call("search_evidence", {"query": f"q{i}"}, f"call-{i}") for i in range(3)] + [
        final({"verified_claim_ids": [], "additional_evidence_ids": ["p1-a"], "reasoning_summary": "done"})]
    ScriptedModelClient.sent = []
    records: list[dict] = []
    model = SummaryModel(model="granite", base_url="http://model", api_key="k", timeout_s=30)
    MCPToolClient(records.append).run_agent(model, "Which attacks?", {"answer_source": "research"})
    checkpoints = [r for r in records if r["tool"] == "context_checkpoint" and r["arguments"].get("mode") == "summary"]
    assert checkpoints and checkpoints[0]["result"]["summarised_exchanges"] >= 2
    assert model.summary_requests[0]["question"] == "Which attacks?"
    last_messages = ScriptedModelClient.sent[-1]
    assert any("Summary of the archived research: searched three times" in str(m.get("content")) for m in last_messages)


def test_scalar_evidence_ids_are_accepted_or_refused_without_crashing(monkeypatch) -> None:
    decision = {"verified_claim_ids": [], "additional_evidence_ids": ["p4-intro"], "reasoning_summary": "done",
                "findings": [{"statement": "VPP is introduced in p4.", "evidence_ids": "p4-intro"}]}
    answer, _session, records = run_loop(monkeypatch, [
        tool_call("search_evidence", {"query": "VPP"}),
        tool_call("record_finding", {"statement": "VPP is introduced in p4.", "evidence_ids": "p4-intro"}, "call-2"),
        tool_call("record_finding", {"statement": "A number.", "evidence_ids": 7}, "call-3"),
        final(decision),
    ])
    assert json.loads(answer) == decision
    statuses = [record["status"] for record in records if record["tool"] == "record_finding"]
    assert statuses == ["ok", "error"]
    assert run_loop.client.notes[0]["evidence_ids"] == ["p4-intro"]


def test_calls_to_tools_not_offered_are_refused_without_mcp(monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_AGENT_MAX_TOOL_CALLS", "1")
    _, session, records = run_loop(monkeypatch, [
        tool_call("search_evidence", {"query": "VPP"}),
        tool_call("search_evidence", {"query": "VPP again"}, "call-2"),
        tool_call("rank_evidence", {"query": "VPP"}, "call-3"),
        final({"verified_claim_ids": [], "additional_evidence_ids": ["p4-intro"], "reasoning_summary": "ok"}),
    ])
    assert [name for name, _ in session.calls] == ["search_evidence"]
    refused = [record for record in records if record["status"] == "unavailable"]
    assert [record["tool"] for record in refused] == ["search_evidence", "rank_evidence"]
    assert "budget" in refused[0]["error"]


def _summary_run(monkeypatch, keep: str, calls: int, text: str = "long passage " * 400) -> tuple[SummaryModel, list[dict]]:
    _session_with(monkeypatch, ["search_evidence"], text=text)
    monkeypatch.setenv("HARNESS_AGENT_CONTEXT_TOKENS", "6000")
    monkeypatch.setenv("HARNESS_AGENT_KEEP_RESULTS", keep)
    session = FakeSession()

    @asynccontextmanager
    async def fake_transport(*args, **kwargs):
        yield (None, None, None)

    monkeypatch.setattr(mcp_module, "streamablehttp_client", fake_transport)
    monkeypatch.setattr(mcp_module, "ClientSession", lambda read, write: session)
    monkeypatch.setattr(tool_loop_module.httpx, "AsyncClient", ScriptedModelClient)
    ScriptedModelClient.script = [tool_call("search_evidence", {"query": f"q{i}"}, f"call-{i}")
                                  for i in range(calls)] + [
        final({"verified_claim_ids": [], "additional_evidence_ids": ["p1-a"], "reasoning_summary": "done"})]
    ScriptedModelClient.sent = []
    records: list[dict] = []
    model = SummaryModel(model="granite", base_url="http://model", api_key="k", timeout_s=30)
    MCPToolClient(records.append).run_agent(model, "Which attacks?", {"answer_source": "research"})
    return model, records


def test_a_new_summary_merges_the_earlier_one(monkeypatch) -> None:
    # Results of ~2,000 characters: a 6,000-token window holds a few exchanges, so summaries recur
    # (larger results, up to a quarter of the window each, are dropped by fitting before a second summary).
    model, _ = _summary_run(monkeypatch, "0", 8, text="long passage " * 150)
    requests = getattr(model, "summary_requests", [])
    assert len(requests) >= 2
    assert requests[0]["earlier_summary"] is None
    assert requests[1]["earlier_summary"].startswith("searched three times")


def test_no_summary_when_older_exchanges_are_already_digests(monkeypatch) -> None:
    # Near the limit (the kept full results are large), but archiving the masked older exchanges
    # would free little: fitting handles it without a model call.
    model, records = _summary_run(monkeypatch, "2", 7)
    assert getattr(model, "summary_requests", []) == []
    assert not [r for r in records if r["tool"] == "context_checkpoint" and r["arguments"].get("mode") == "summary"]


def test_a_slow_summary_does_not_outlive_the_agent_phase(monkeypatch) -> None:
    import time

    import pytest
    from harness_extraction import run_interruptible

    class SlowSummaryModel(SummaryModel):
        def complete_json(self, system: str, user: str) -> dict:
            return run_interruptible(lambda: time.sleep(10) or {"summary": "late"})

    _session_with(monkeypatch, ["search_evidence"], text="long passage " * 400)
    monkeypatch.setenv("HARNESS_AGENT_CONTEXT_TOKENS", "6000")
    monkeypatch.setenv("HARNESS_AGENT_KEEP_RESULTS", "0")
    monkeypatch.setenv("HARNESS_MCP_AGENT_TIMEOUT_S", "2")
    session = FakeSession()

    @asynccontextmanager
    async def fake_transport(*args, **kwargs):
        yield (None, None, None)

    monkeypatch.setattr(mcp_module, "streamablehttp_client", fake_transport)
    monkeypatch.setattr(mcp_module, "ClientSession", lambda read, write: session)
    monkeypatch.setattr(tool_loop_module.httpx, "AsyncClient", ScriptedModelClient)
    ScriptedModelClient.script = [tool_call("search_evidence", {"query": f"q{i}"}, f"call-{i}") for i in range(4)] + [
        final({"verified_claim_ids": [], "additional_evidence_ids": ["p1-a"], "reasoning_summary": "done"})]
    model = SlowSummaryModel(model="granite", base_url="http://model", api_key="k", timeout_s=30)
    started = time.monotonic()
    with pytest.raises(mcp_module.MCPUnavailable):
        MCPToolClient(lambda record: None).run_agent(model, "Which attacks?", {"answer_source": "research"})
    assert time.monotonic() - started < 7.5   # phase 2 s + grace 2 s, not the 10 s call


def test_the_scratchpad_alone_is_never_forced(monkeypatch) -> None:
    # Only record_finding on offer (no corpus tool listed): tool_choice must not be "required".
    _session_with(monkeypatch, [])
    ScriptedModelClient.bodies = []
    run_loop(monkeypatch, [final({"verified_claim_ids": [], "additional_evidence_ids": [], "reasoning_summary": "x"}),
                           final({"verified_claim_ids": [], "additional_evidence_ids": [], "reasoning_summary": "x"})])
    first = ScriptedModelClient.bodies[0]
    assert [tool["function"]["name"] for tool in first["tools"]] == ["record_finding"]
    assert first["tool_choice"] == "auto"


def parallel_calls(*calls: tuple[str, dict]) -> dict:
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": f"par-{index}", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
        for index, (name, arguments) in enumerate(calls)]}


def test_the_tool_budget_is_a_hard_cap_per_call_even_for_parallel_calls(monkeypatch) -> None:
    # Budget 3: one call, then five in parallel (four corpus calls and a finding) -> only two more run.
    monkeypatch.setenv("HARNESS_AGENT_MAX_TOOL_CALLS", "3")
    _session_with(monkeypatch, ["search_evidence"], text="a passage")
    ScriptedModelClient.bodies = []
    _, session, records = run_loop(monkeypatch, [
        tool_call("search_evidence", {"query": "q0"}),
        parallel_calls(*[("search_evidence", {"query": f"q{n}"}) for n in range(1, 5)],
                       ("record_finding", {"statement": "p1-a matters.", "evidence_ids": ["p1-a"]})),
        final({"verified_claim_ids": [], "additional_evidence_ids": ["p1-a"], "reasoning_summary": "ok"}),
    ])
    assert len(session.calls) == 3                                   # never more than the budget
    corpus = [record for record in records if record["tool"] == "search_evidence"]
    assert [record["status"] for record in corpus] == ["ok", "ok", "ok", "unavailable", "unavailable"]
    assert corpus[3]["error"].startswith("unavailable: tool budget reached (3 corpus calls)")
    finding = [record for record in records if record["tool"] == "record_finding"]
    assert [record["status"] for record in finding] == ["ok"]        # the scratchpad stays available
    third = ScriptedModelClient.bodies[2]
    assert [tool["function"]["name"] for tool in third["tools"]] == ["record_finding"]
    refused = [m for m in ScriptedModelClient.sent[2] if m.get("tool_call_id") in {"par-2", "par-3"}]
    assert len(refused) == 2 and all("tool budget reached" in m["content"] for m in refused)


def test_the_last_allowed_turn_offers_no_tools_and_asks_for_the_decision(monkeypatch) -> None:
    # Budget 1 -> 1 + TURN_MARGIN turns; an agent that keeps recording findings is stopped there.
    monkeypatch.setenv("HARNESS_AGENT_MAX_TOOL_CALLS", "1")
    max_turns = 1 + tool_loop_module.TURN_MARGIN
    decision = {"verified_claim_ids": [], "additional_evidence_ids": ["p4-intro"], "reasoning_summary": "ok"}
    ScriptedModelClient.bodies = []
    answer, _, records = run_loop(monkeypatch, [
        tool_call("search_evidence", {"query": "VPP"}),
        *[tool_call("record_finding", {"statement": f"note {n}", "evidence_ids": ["p4-intro"]}, f"call-{n}")
          for n in range(max_turns - 2)],
        final(decision),
    ])
    assert json.loads(answer) == decision
    assert len(ScriptedModelClient.bodies) == max_turns
    assert [r["status"] for r in records if r["tool"] == "record_finding"] == ["ok"] * (max_turns - 2)
    last = ScriptedModelClient.bodies[-1]
    assert "tools" not in last and "last turn" in last["messages"][-1]["content"]


def test_tool_calls_on_the_last_turn_end_the_loop_like_the_deadline(monkeypatch) -> None:
    import pytest

    monkeypatch.setenv("HARNESS_AGENT_MAX_TOOL_CALLS", "1")
    max_turns = 1 + tool_loop_module.TURN_MARGIN
    with pytest.raises(mcp_module.MCPUnavailable, match="turn limit"):
        run_loop(monkeypatch, [
            tool_call("search_evidence", {"query": "VPP"}),
            *[tool_call("search_evidence", {"query": "VPP"}, f"call-{n}") for n in range(max_turns - 1)],
        ])
    assert ScriptedModelClient.script == []                          # exactly max_turns turns were sent


def test_turn_limit_is_recorded_as_a_checkpoint(monkeypatch) -> None:
    import pytest

    monkeypatch.setenv("HARNESS_AGENT_MAX_TOOL_CALLS", "1")
    max_turns = 1 + tool_loop_module.TURN_MARGIN
    session = FakeSession()

    @asynccontextmanager
    async def fake_transport(*args, **kwargs):
        yield (None, None, None)

    monkeypatch.setattr(mcp_module, "streamablehttp_client", fake_transport)
    monkeypatch.setattr(mcp_module, "ClientSession", lambda read, write: session)
    monkeypatch.setattr(tool_loop_module.httpx, "AsyncClient", ScriptedModelClient)
    ScriptedModelClient.script = [tool_call("search_evidence", {"query": "VPP"}, f"call-{n}") for n in range(max_turns)]
    records: list[dict] = []
    model = SimpleNamespace(model="granite", base_url="http://model", api_key="k", timeout_s=30)
    with pytest.raises(mcp_module.MCPUnavailable):
        MCPToolClient(records.append).run_agent(model, "Which paper introduced VPP?", {"answer_source": "research"})
    [stop] = [r for r in records if r["tool"] == "context_checkpoint" and r["arguments"].get("mode") == "turn_limit"]
    assert stop["result"]["turns"] == max_turns and stop["result"]["corpus_calls"] == 1
    assert len(session.calls) == 1                                   # the later calls were refused or cached


def test_max_size_results_and_history_are_compacted_not_overflowed(monkeypatch) -> None:
    # Window 12,000 tokens: usable limit (12,000 - 2,048) * 0.9 = 8,956; one result may take a quarter
    # (2,239 tokens = 6,717 characters). Three such results per turn over several turns exceed the window.
    from harness_agent.context import estimate_tokens

    _session_with(monkeypatch, ["search_evidence"], text="long passage " * 4000)
    monkeypatch.setenv("HARNESS_AGENT_SUMMARY_COMPACTION", "false")
    ScriptedModelClient.bodies = []
    script = [parallel_calls(*[("search_evidence", {"query": f"q{turn}-{n}"}) for n in range(3)]) for turn in range(4)]
    _, session, records = run_loop(monkeypatch, [*script, final(
        {"verified_claim_ids": [], "additional_evidence_ids": ["p1-a"], "reasoning_summary": "ok"})], window="12000")
    limit = int((12_000 - 2_048) * 0.9)
    first_results = [m for m in ScriptedModelClient.bodies[1]["messages"] if m["role"] == "tool"]
    assert len(first_results) == 3
    assert all(6_717 <= len(m["content"]) <= 6_717 + 100 and "truncated" in m["content"] for m in first_results)
    for body in ScriptedModelClient.bodies:
        assert estimate_tokens(body["messages"]) + estimate_tokens(body.get("tools", [])) <= limit
    assert [r for r in records if r["tool"] == "context_checkpoint"]   # compaction ran
    assert len(session.calls) == 12
