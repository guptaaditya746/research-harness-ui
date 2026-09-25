"""Synchronous bridge from the threaded API to the Streamable HTTP MCP server."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from typing import Any

import httpx
from harness_config import settings
from harness_extraction import ModelFailure, bounded_timeout, call_interrupted, emit_model_event
from harness_prompts import render, text
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from .context import (
    Conversation,
    cap_tool_result,
    discover_window,
    is_context_error,
    window_from_error,
)
from .research_state import ResearchLedger, _handles
from .skills import catalog, relevant

ToolSink = Callable[[dict[str, Any]], None]


def _intent(name: str) -> tuple[str, str]:
    evidence_type = {
        "search_evidence": "ranked_passage", "read_evidence": "passage",
        "structured_facts": "canonical_metadata", "evidence_for_claim": "claim_evidence",
        "facts": "canonical_metadata", "citations": "citation_fact", "search": "passage",
        "passages": "passage", "tables": "table_cell", "rank_evidence": "ranked_passage",
        "prior_triples": "claim_subgraph", "related": "related_entity",
    }.get(name, "corpus_evidence")
    return (f"Collect {evidence_type.replace('_', ' ')} relevant to the active research question.", evidence_type)


def _serialise(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, (list, tuple)):
        return [_serialise(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serialise(item) for key, item in value.items()}
    return value


def _tool_payload(result: Any) -> Any:
    """Unwrap the MCP text envelope before passing data back to the model."""
    if not isinstance(result, dict) or not isinstance(result.get("content"), list):
        return result
    texts = [
        item["text"] for item in result["content"]
        if isinstance(item, dict) and isinstance(item.get("text"), str)
    ]
    if len(texts) != 1:
        return result
    try:
        return json.loads(texts[0])
    except json.JSONDecodeError:
        return {"text": texts[0]}


DECISION_KEYS = ("verified_claim_ids", "additional_evidence_ids", "reasoning_summary")


def decision_problems(
    final: str, *, observed: set[str], claim_ids: set[str], corpus_calls: int, needs_tools: bool,
) -> list[str]:
    """Why the agent's final decision cannot be accepted as-is (empty when it can).

    Checked deterministically before the decision leaves the tool loop, so the
    agent gets one chance to repair it instead of the run degrading.
    """
    start, end = final.find("{"), final.rfind("}")
    try:
        decision = json.loads(final[start:end + 1]) if start >= 0 else None
    except json.JSONDecodeError:
        decision = None
    if not isinstance(decision, dict):
        return ["the final message must be one JSON object with keys " + ", ".join(DECISION_KEYS)]
    problems: list[str] = []
    for key in ("verified_claim_ids", "additional_evidence_ids"):
        if not isinstance(decision.get(key), list):
            problems.append(f"{key} must be a list (use [] when empty)")
    evidence = [str(item) for item in decision.get("additional_evidence_ids") or [] if isinstance(decision.get("additional_evidence_ids"), list)]
    unobserved = [item for item in evidence if item not in observed]
    if unobserved:
        problems.append(
            "these additional_evidence_ids were never returned by any tool in this session and must be "
            "removed or replaced by ids you actually read: " + ", ".join(unobserved[:8])
        )
    verified = [str(item) for item in decision.get("verified_claim_ids") or [] if isinstance(decision.get("verified_claim_ids"), list)]
    unknown = [item for item in verified if item not in claim_ids]
    if unknown:
        problems.append(
            "these verified_claim_ids are not preliminary claim ids (allowed: "
            + ", ".join(sorted(claim_ids)[:8]) + "): " + ", ".join(unknown[:8])
        )
    if needs_tools and corpus_calls == 0:
        problems.append("no corpus tool was called; gather evidence with at least one tool call before deciding")
    return problems


class MCPUnavailable(RuntimeError):
    """Raised when the configured managed MCP server cannot be reached."""


class MCPToolClient:
    """Call one managed MCP tool per isolated session.

    The API runs runners in ordinary worker threads, while the MCP SDK is
    asynchronous.  Creating a short-lived async session per call keeps the
    boundary explicit and avoids sharing an event loop across run threads.
    """

    def __init__(self, sink: ToolSink | None = None, *, agent: str = "supervisor") -> None:
        self.url = settings().mcp_corpus_url
        self.prompt_ref: str | None = None
        self.sink = sink or (lambda _: None)
        self.agent = agent
        self._counter = 0

    async def _call_async(self, name: str, arguments: dict[str, Any]) -> Any:
        async with streamablehttp_client(
            self.url, timeout=30, sse_read_timeout=300, terminate_on_close=False
        ) as (
            read_stream,
            write_stream,
            _,
        ), ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(name, arguments)
            return _serialise(result)

    async def _call_many_async(self, calls: list[tuple[str, dict[str, Any], str]]) -> list[Any]:
        async with streamablehttp_client(self.url, timeout=30, sse_read_timeout=300) as (
            read_stream,
            write_stream,
            _,
        ), ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            return [
                _serialise(await session.call_tool(name, arguments))
                for name, arguments, _ in calls
            ]

    async def _run_agent_async(
        self, model: Any, question: str, grounded_context: dict[str, Any] | None = None
    ) -> str:
        async with streamablehttp_client(self.url, timeout=30, sse_read_timeout=300) as (
            read_stream,
            write_stream,
            _,
        ), ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            listed = await session.list_tools()
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": item.name,
                        "description": item.description or "",
                        "parameters": item.inputSchema,
                    },
                }
                for item in listed.tools
            ]
            ledger = ResearchLedger(question=question)
            loaded_skills = relevant(question)
            for skill in loaded_skills:
                self._counter += 1
                self.sink({
                    "call_id": f"skill-{self._counter:03d}", "agent": self.agent,
                    "tool": "load_skill", "arguments": {"name": skill.name, "version": skill.version},
                    "result": {"name": skill.name, "version": skill.version}, "result_count": 1,
                    "duration_ms": 0.0, "status": "ok",
                    "purpose": "Load a reusable research strategy for the active question.",
                    "expected_evidence_type": "research_skill",
                })
            supervisor = render(
                "agent/research_supervisor",
                skills_catalog=catalog(),
                loaded_skills="; ".join(
                    f"{skill.name}@{skill.version}: {skill.instructions}" for skill in loaded_skills),
                skills=loaded_skills,
            )
            self.prompt_ref = supervisor.ref
            fixed: list[dict[str, Any]] = [
                {"role": "system", "content": supervisor.system},
                {"role": "user", "content": question},
            ]
            conversation = Conversation(fixed, grounded=grounded_context or None)
            # Priority order: the evidence tools the supervisor is told to prefer,
            # then the specialist tools, which are dropped first when space is short.
            core_names = ["search_evidence", "read_evidence", "structured_facts", "aggregate_facts",
                          "metadata_table", "citations", "evidence_for_claim"]
            tools.sort(key=lambda tool: (
                core_names.index(tool["function"]["name"])
                if tool["function"]["name"] in core_names else len(core_names)
            ))
            core_count = sum(1 for tool in tools if tool["function"]["name"] in core_names)
            all_tools = list(tools)
            window = discover_window(model)
            configured_reply = settings().agent_reply_tokens
            reply_tokens = min(configured_reply, max(512, window // 4))
            retried_for_length = False
            turn_number = 0
            # One bounded self-correction: a decision that fails validation is
            # sent back once with the exact problems, instead of degrading the run.
            reflections_left = settings().agent_reflections
            shown_ids = _handles(grounded_context or {}) | {
                str(claim.get("id")) for claim in (grounded_context or {}).get("preliminary_claims", [])
                if isinstance(claim, dict) and claim.get("id") is not None
            }
            claim_ids = {
                str(claim.get("id")) for claim in (grounded_context or {}).get("preliminary_claims", [])
                if isinstance(claim, dict) and claim.get("id") is not None
            }
            needs_tools = (grounded_context or {}).get("answer_source") != "verified_facts"

            def limit() -> int:
                # Leave room for the reply and a margin for estimation error.
                return max(2_048, int((window - reply_tokens) * 0.9))

            def compact() -> None:
                before = conversation.tokens(tools)
                if conversation.fit(limit(), tools, ledger.summary(), core_tools=core_count):
                    self._counter += 1
                    self.sink({
                        "call_id": f"context-{self._counter:03d}", "agent": self.agent,
                        "tool": "context_checkpoint", "arguments": {},
                        "result": {"window": window, "before_tokens": before,
                                   "after_tokens": conversation.tokens(tools),
                                   "tools_offered": len(tools), "tools_available": len(all_tools),
                                   "archived_exchanges": conversation.dropped_exchanges,
                                   "evidence_ids": sorted(ledger.evidence_ids)},
                        "result_count": len(ledger.evidence_ids), "duration_ms": 0.0, "status": "ok",
                        "purpose": "Keep the conversation within the model context window.",
                        "expected_evidence_type": "research_ledger",
                    })

            async with httpx.AsyncClient(timeout=model.timeout_s) as client:
                while True:
                    try:
                        turn_timeout = bounded_timeout(float(model.timeout_s))
                    except ModelFailure:
                        # Stopped or out of time: end the loop with what we have.
                        return ""
                    compact()
                    needed = conversation.tokens(tools)
                    if needed > limit():
                        raise ModelFailure(
                            f"agent context window ({window} tokens) is too small: the minimal "
                            f"prompt with {len(tools)} core tools needs about {needed + reply_tokens} "
                            "tokens; serve the agent model with a larger max_model_len or set "
                            "HARNESS_AGENT_CONTEXT_TOKENS"
                        )
                    turn_number += 1
                    turn_started = time.monotonic()
                    alias = getattr(model, "model", "agent")
                    emit_model_event(
                        "Model call started",
                        f"{alias} · agent tool loop turn {turn_number} · {len(tools)} tools",
                    )
                    turn = asyncio.ensure_future(client.post(
                        f"{model.base_url}/chat/completions",
                        timeout=turn_timeout,
                        headers={"Authorization": f"Bearer {model.api_key}"},
                        json={
                            "model": getattr(model, "model", ""),
                            "temperature": 0,
                            "max_tokens": reply_tokens,
                            "messages": conversation.messages(),
                            "tools": tools,
                            "tool_choice": "auto",
                        },
                    ))
                    # Wait in short intervals so Stop or the run deadline can
                    # abandon a slow model turn immediately.
                    while not turn.done():
                        await asyncio.wait({turn}, timeout=0.25)
                        if not turn.done() and call_interrupted():
                            turn.cancel()
                            return ""
                    try:
                        response = turn.result()
                    except httpx.TimeoutException:
                        emit_model_event("Model call failed", f"{alias} · agent tool loop turn "
                                         f"{turn_number} · timed out after {time.monotonic() - turn_started:.0f}s")
                        raise
                    if response.status_code >= 400:
                        body = response.text[:600]
                        emit_model_event("Model call failed", f"{alias} · agent tool loop turn "
                                         f"{turn_number} · HTTP {response.status_code}: {body[:200]}")
                        if is_context_error(body) and not retried_for_length:
                            # Learn the real window from the provider and refit once.
                            retried_for_length = True
                            learned = window_from_error(body)
                            window = min(window, learned) if learned else int(window * 0.6)
                            reply_tokens = min(configured_reply, max(512, window // 4))
                            continue
                        raise ModelFailure(f"model returned HTTP {response.status_code}: {body}")
                    message = response.json()["choices"][0]["message"]
                    requested = message.get("tool_calls") or []
                    emit_model_event(
                        "Model call finished",
                        f"{alias} · agent tool loop turn {turn_number} · "
                        f"{time.monotonic() - turn_started:.1f}s · "
                        + (f"{len(requested)} tool calls" if requested else "final answer")
                        + (f" · prompt {self.prompt_ref}" if turn_number == 1 and self.prompt_ref else ""),
                    )
                    if not requested:
                        final = str(message.get("content") or "")
                        problems = decision_problems(
                            final, observed=shown_ids | ledger.evidence_ids, claim_ids=claim_ids,
                            corpus_calls=ledger.tool_calls, needs_tools=needs_tools,
                        )
                        if problems and reflections_left > 0:
                            reflections_left -= 1
                            emit_model_event("Agent decision sent back", "; ".join(problems)[:400])
                            conversation.add_exchange(
                                {"role": "assistant", "content": final},
                                [{"role": "user", "content": text(
                                    "agent/decision_rejected", problems=problems)}],
                            )
                            continue
                        return final
                    assistant = {
                        "role": "assistant",
                        "content": message.get("content"),
                        "tool_calls": requested,
                    }
                    tool_messages: list[dict[str, Any]] = []
                    for call in requested:
                        name = call["function"]["name"]
                        started = time.monotonic()
                        status = "ok"
                        error: str | None = None
                        try:
                            arguments = json.loads(call["function"].get("arguments") or "{}")
                            if not isinstance(arguments, dict):
                                raise TypeError("tool arguments must be a JSON object")
                        except (json.JSONDecodeError, TypeError) as exc:
                            # A malformed call is reported back to the model as
                            # that tool's result so it can correct itself.
                            arguments = {"_raw": str(call["function"].get("arguments"))[:500]}
                            result = {"error": f"invalid tool arguments: {exc}"}
                            status = "error"
                            error = result["error"]
                        else:
                            repeat = ledger.repeat_of(name, arguments)
                        if error:
                            pass
                        elif repeat is not None:
                            # The same call again returns the same data; answer
                            # from the ledger and say so, instead of looping.
                            earlier_id, earlier_result = repeat
                            result = {
                                "repeated_call": earlier_id,
                                "note": "identical to an earlier call; the corpus has not changed. Use the "
                                        "ids you already have, change the arguments, or decide.",
                                "result": earlier_result,
                            }
                            status = "repeat"
                        else:
                            try:
                                raw = await session.call_tool(name, arguments)
                                result = _tool_payload(_serialise(raw))
                            except Exception as exc:  # noqa: BLE001
                                result = {"error": str(exc)}
                                status = "error"
                                error = str(exc)
                            hint = ledger.loop_hint(name, arguments)
                            if hint and isinstance(result, dict):
                                result = {**result, "hint": hint}
                                emit_model_event("Tool-loop hint", f"{name} called for several papers; aggregate tools suggested")
                        self._counter += 1
                        record = {
                            "call_id": f"tool-{self._counter:03d}",
                            "agent": self.agent,
                            "tool": name,
                            "arguments": arguments,
                            "result": result,
                            "result_count": self._result_count(result),
                            "duration_ms": round((time.monotonic() - started) * 1000, 1),
                            "status": status,
                        }
                        record["purpose"], record["expected_evidence_type"] = _intent(name)
                        if error:
                            record["error"] = error
                        self.sink(record)
                        # Live progress: which tool ran and what it returned.
                        emit_model_event("Tool call", f"{name} · " + (
                            f"error: {str(error)[:160]}" if error
                            else "repeated call answered from the ledger" if status == "repeat"
                            else "returned" if record["result_count"] is None
                            else f"{record['result_count']} results"
                        ) + f" · {record['duration_ms']:.0f} ms")
                        if status != "repeat":
                            ledger.record(name, result, record["purpose"], arguments=arguments,
                                          call_id=record["call_id"])
                        tool_messages.append({
                            "role": "tool",
                            "tool_call_id": call["id"],
                            # The full result is kept in tool_calls.jsonl; the
                            # model sees a page bounded by the window.
                            "content": cap_tool_result(
                                json.dumps(result, ensure_ascii=False, default=str), limit() // 4
                            ),
                        })
                    # An exchange is only added whole: the assistant request
                    # together with one result per requested call.
                    conversation.add_exchange(assistant, tool_messages)

    def run_agent(
        self, model: Any, question: str, grounded_context: dict[str, Any] | None = None
    ) -> str:
        """Run an open-ended model/tool conversation over one MCP session."""
        try:
            # The supervisor phase has its own deadline. A slow model turn
            # must not consume the entire research run; observations already
            # returned by tools can still seed the shared pipeline.
            timeout_s = settings().mcp_agent_timeout_s
            # Never outlive the run itself.
            timeout_s = bounded_timeout(timeout_s)
            return asyncio.run(asyncio.wait_for(
                self._run_agent_async(model, question, grounded_context), timeout_s
            ))
        except Exception as exc:
            def detail(error: BaseException) -> str:
                children = getattr(error, "exceptions", ())
                return "; ".join(detail(child) for child in children) if children else f"{type(error).__name__}: {error}"

            raise MCPUnavailable(f"MCP agent workflow failed: {detail(exc)}") from exc

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        args = arguments or {}
        self._counter += 1
        call_id = f"tool-{self._counter:03d}"
        started = time.monotonic()
        status = "ok"
        result: Any = None
        error: str | None = None
        try:
            result = asyncio.run(self._call_async(name, args))
        except Exception as exc:
            status = "error"
            error = str(exc)
            raise MCPUnavailable(f"MCP tool {name} failed: {exc}") from exc
        finally:
            elapsed = round((time.monotonic() - started) * 1000, 1)
            record = {
                "call_id": call_id,
                "agent": self.agent,
                "tool": name,
                "arguments": args,
                "result": result,
                "result_count": self._result_count(result),
                "duration_ms": elapsed,
                "status": status,
            }
            record["purpose"], record["expected_evidence_type"] = _intent(name)
            if error:
                record["error"] = error
            self.sink(record)
        return result

    def call_many(self, calls: list[tuple[str, dict[str, Any], str]]) -> list[Any]:
        """Execute a workflow's calls over one MCP session."""
        started = time.monotonic()
        results: list[Any] = []
        try:
            results = asyncio.run(self._call_many_async(calls))
        except Exception as exc:
            raise MCPUnavailable(f"MCP workflow failed: {exc}") from exc
        for (name, arguments, agent), result in zip(calls, results, strict=True):
            self._counter += 1
            record = {
                "call_id": f"tool-{self._counter:03d}",
                "agent": agent,
                "tool": name,
                "arguments": arguments,
                "result": result,
                "result_count": self._result_count(result),
                "duration_ms": round((time.monotonic() - started) * 1000, 1),
                "status": "ok",
            }
            record["purpose"], record["expected_evidence_type"] = _intent(name)
            self.sink(record)
        return results

    @staticmethod
    def _result_count(result: Any) -> int | None:
        if isinstance(result, dict):
            if isinstance(result.get("items"), list):
                return len(result["items"])
            content = result.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        try:
                            decoded = json.loads(item["text"])
                        except json.JSONDecodeError:
                            continue
                        if isinstance(decoded, dict) and isinstance(decoded.get("items"), list):
                            return len(decoded["items"])
                return len(content)
        if isinstance(result, list):
            return len(result)
        return None


def tool_preview(result: Any) -> str:
    """Compact text used in trace rows and model context."""
    if isinstance(result, dict) and isinstance(result.get("items"), list):
        return json.dumps(
            {"total": result.get("total"), "returned": len(result["items"])},
            separators=(",", ":"),
        )
    text = json.dumps(result, ensure_ascii=False, default=str)
    return text if len(text) <= 500 else text[:500] + "…"


__all__ = ["MCPToolClient", "MCPUnavailable", "tool_preview"]
