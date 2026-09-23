"""Synchronous bridge from the threaded API to the Streamable HTTP MCP server."""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Callable
from typing import Any

import httpx
from harness_extraction import ModelFailure, bounded_timeout, call_interrupted
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from .research_state import ResearchLedger
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


class MCPUnavailable(RuntimeError):
    """Raised when the configured managed MCP server cannot be reached."""


class MCPToolClient:
    """Call one managed MCP tool per isolated session.

    The API runs runners in ordinary worker threads, while the MCP SDK is
    asynchronous.  Creating a short-lived async session per call keeps the
    boundary explicit and avoids sharing an event loop across run threads.
    """

    def __init__(self, sink: ToolSink | None = None, *, agent: str = "supervisor") -> None:
        self.url = os.getenv("MCP_CORPUS_URL", "http://127.0.0.1:8765/mcp")
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
            messages: list[dict[str, Any]] = [
                {
                    "role": "system",
                    "content": (
                        "You are the research supervisor. Form hypotheses, then freely choose corpus "
                        "MCP evidence tools. Prefer search_evidence, read_evidence, structured_facts, "
                        "citations, tables, and evidence_for_claim; use specialist tools when needed. "
                        "Preliminary claims are "
                        "candidates, not authority. Verify or challenge them and follow additional "
                        "evidence when it could materially change the answer. Stop when the answer "
                        "requirements are evidenced or a complete indexed absence procedure is "
                        "exhausted. Return only JSON with verified_claim_ids (list of preliminary "
                        "claim IDs), additional_evidence_ids (list of observed passage IDs), and "
                        "reasoning_summary. Never invent IDs, facts, or counts. Available skills: "
                        + catalog() + ". Loaded skills: "
                        + "; ".join(f"{skill.name}@{skill.version}: {skill.instructions}" for skill in loaded_skills)
                    ),
                },
                {"role": "user", "content": question},
            ]
            if grounded_context:
                messages.append({
                    "role": "user",
                    "content": (
                        "Preliminary corpus result (verify or challenge with MCP). "
                        "Source IDs are provenance handles, not paper IDs. When a tool needs paper_id, "
                        "use the corresponding local paper_id from evidence_map. For corpus aggregation, "
                        "retrieve every indexed paper rather than one default page: "
                        + json.dumps(grounded_context, ensure_ascii=False)
                    ),
                })
            async with httpx.AsyncClient(timeout=model.timeout_s) as client:
                while True:
                    try:
                        turn_timeout = bounded_timeout(float(model.timeout_s))
                    except ModelFailure:
                        # Stopped or out of time: end the loop with what we have.
                        return ""
                    turn = asyncio.ensure_future(client.post(
                        f"{model.base_url}/chat/completions",
                        timeout=turn_timeout,
                        headers={"Authorization": f"Bearer {model.api_key}"},
                        json={
                            "model": getattr(model, "model", ""),
                            "temperature": 0,
                            "max_tokens": 1024,
                            "messages": messages,
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
                    response = turn.result()
                    response.raise_for_status()
                    message = response.json()["choices"][0]["message"]
                    requested = message.get("tool_calls") or []
                    if not requested:
                        return str(message.get("content") or "")
                    messages.append(
                        {
                            "role": "assistant",
                            "content": message.get("content"),
                            "tool_calls": requested,
                        }
                    )
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
                            try:
                                raw = await session.call_tool(name, arguments)
                                result = _tool_payload(_serialise(raw))
                            except Exception as exc:  # noqa: BLE001
                                result = {"error": str(exc)}
                                status = "error"
                                error = str(exc)
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
                        ledger.record(name, result, record["purpose"])
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call["id"],
                                "content": json.dumps(result, ensure_ascii=False, default=str),
                            }
                        )
                        checkpoint = ledger.checkpoint(messages)
                        if checkpoint:
                            self._counter += 1
                            self.sink({
                                "call_id": f"context-{self._counter:03d}", "agent": self.agent,
                                "tool": "context_checkpoint", "arguments": {}, "result": checkpoint,
                                "result_count": len(ledger.evidence_ids), "duration_ms": 0.0, "status": "ok",
                                "purpose": "Preserve research state and exact evidence handles while compacting context.",
                                "expected_evidence_type": "research_ledger",
                            })
                            messages = messages[:2] + [{
                                "role": "user",
                                "content": "Research ledger checkpoint (raw evidence remains available by ID): " + ledger.summary(),
                            }] + messages[-4:]

    def run_agent(
        self, model: Any, question: str, grounded_context: dict[str, Any] | None = None
    ) -> str:
        """Run an open-ended model/tool conversation over one MCP session."""
        try:
            # The tool loop must outlast at least two model turns, otherwise a
            # single slow turn discards everything gathered so far.
            per_turn = float(getattr(model, "timeout_s", 120) or 120)
            timeout_s = max(float(os.getenv("HARNESS_MCP_AGENT_TIMEOUT_S", "240")), 2 * per_turn)
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
