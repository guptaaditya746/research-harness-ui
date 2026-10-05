"""Synchronous bridge from the threaded API to the Streamable HTTP MCP server."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from typing import Any

from harness_config import settings
from harness_extraction import bounded_timeout, phase_budget
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

# Re-exported: callers and tests import these from here.
from .decisions import (  # noqa: F401
    CURATED_TOOLS,
    DECISION_KEYS,
    FINDING_TOOL,
    FINDING_TOOL_SPEC,
    decision_problems,
    id_list,
    validate_findings,
)
from .tool_loop import ToolLoop, intent, result_count, serialise
from .tool_loop import tool_payload as _tool_payload  # noqa: F401

PHASE_GRACE_S = 2.0   # wait_for ends the phase first; threads are abandoned this much later
ToolSink = Callable[[dict[str, Any]], None]


def _first_non_exception(group: BaseExceptionGroup) -> BaseException | None:
    """The first leaf of an exception group that is not an ``Exception`` (e.g. a cancellation)."""
    for error in group.exceptions:
        if isinstance(error, BaseExceptionGroup):
            found = _first_non_exception(error)
            if found is not None:
                return found
        elif not isinstance(error, Exception):
            return error
    return None


def _detail(error: BaseException) -> str:
    children = getattr(error, "exceptions", ())
    return "; ".join(_detail(child) for child in children) if children else f"{type(error).__name__}: {error}"


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
        # The agent's findings, kept even when the loop ends early (timeout, Stop).
        self.notes: list[dict[str, Any]] = []

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
            return serialise(result)

    async def _call_many_async(self, calls: list[tuple[str, dict[str, Any], str]]) -> list[Any]:
        async with streamablehttp_client(self.url, timeout=30, sse_read_timeout=300) as (
            read_stream,
            write_stream,
            _,
        ), ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            return [
                serialise(await session.call_tool(name, arguments))
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
            return await ToolLoop(self, model, session, question, grounded_context).run()

    def next_id(self, prefix: str) -> str:
        """The next call id of this client's records ("tool-007", "context-008", ...)."""
        self._counter += 1
        return f"{prefix}-{self._counter:03d}"

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
            # Blocking model calls inside the loop (the context summary runs in a thread) are
            # abandoned shortly after the phase deadline too; wait_for alone cannot stop a thread,
            # and asyncio.run would wait for it.
            with phase_budget(timeout_s + PHASE_GRACE_S):
                return asyncio.run(asyncio.wait_for(
                    self._run_agent_async(model, question, grounded_context), timeout_s
                ))
        except BaseExceptionGroup as group:
            # The MCP client runs in an anyio task group, so a run's Stop or
            # deadline (RunCancelled, a BaseException raised by the trace sink)
            # arrives wrapped in a group. Re-raise it so the runtime finalizes
            # the run as stopped; otherwise the worker thread dies silently.
            cancelled = _first_non_exception(group)
            if cancelled is not None:
                raise cancelled from None
            raise MCPUnavailable(f"MCP agent workflow failed: {_detail(group)}") from group
        except Exception as exc:
            def detail(error: BaseException) -> str:
                children = getattr(error, "exceptions", ())
                return "; ".join(detail(child) for child in children) if children else f"{type(error).__name__}: {error}"

            raise MCPUnavailable(f"MCP agent workflow failed: {detail(exc)}") from exc

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        args = arguments or {}
        call_id = self.next_id("tool")
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
                "result_count": result_count(result),
                "duration_ms": elapsed,
                "status": status,
            }
            record["purpose"], record["expected_evidence_type"] = intent(name)
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
            record = {
                "call_id": self.next_id("tool"),
                "agent": agent,
                "tool": name,
                "arguments": arguments,
                "result": result,
                "result_count": result_count(result),
                "duration_ms": round((time.monotonic() - started) * 1000, 1),
                "status": "ok",
            }
            record["purpose"], record["expected_evidence_type"] = intent(name)
            self.sink(record)
        return results


def tool_preview(result: Any) -> str:
    """Compact text used in trace rows and model context."""
    if isinstance(result, dict) and isinstance(result.get("items"), list):
        return json.dumps(
            {"total": result.get("total"), "returned": len(result["items"])},
            separators=(",", ":"),
        )
    text = json.dumps(result, ensure_ascii=False, default=str)
    return text if len(text) <= 500 else text[:500] + "…"


__all__ = ["FINDING_TOOL", "MCPToolClient", "MCPUnavailable", "id_list", "tool_preview", "validate_findings"]
