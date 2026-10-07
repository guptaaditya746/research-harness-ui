"""The research agent's tool loop over one open MCP session.

``MCPToolClient._run_agent_async`` opens the session and runs ``ToolLoop(...).run()``. Each turn:
choose the tools on offer (budget, forced first call), keep the conversation within the window
(summary, then fitting), send the turn, then either check the final decision (sent back once with
its problems) or run the requested tool calls and add the exchange.

Bounds: the tool budget (``HARNESS_AGENT_MAX_TOOL_CALLS``) is a hard cap per call, so parallel calls
in one turn cannot overshoot it; the number of turns is capped at the budget plus ``TURN_MARGIN``
(the last turn offers no tools and asks for the decision); the phase deadline is ``mcp.py``'s.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from typing import Any

import httpx
from harness_config import settings
from harness_extraction import (
    ModelFailure,
    bounded_timeout,
    call_interrupted,
    emit_model_event,
    using_prompt,
)
from harness_prompts import render, text

from .context import (
    Conversation,
    cap_tool_result,
    discover_window,
    is_context_error,
    result_digest,
    window_from_error,
)
from .decisions import (
    CURATED_TOOLS,
    FINDING_TOOL,
    FINDING_TOOL_SPEC,
    decision_problems,
    id_list,
    validate_findings,
)
from .research_state import ResearchLedger, _handles
from .skills import catalog, relevant

ToolSink = Callable[[dict[str, Any]], None]
SUMMARISE_AT = 0.8     # share of the usable window at which older exchanges are summarised
SUMMARY_MIN_SAVING = 1_500  # tokens the archivable exchanges must hold before a summary is worth a call
# Turns allowed beyond the tool budget: findings, the budget note, a sent-back decision, refused or
# repeated calls. Without a tool budget (0) only the phase deadline bounds the loop.
TURN_MARGIN = 8
# Priority order: the evidence tools the supervisor is told to prefer, then the specialist tools,
# which are dropped first when space is short.
CORE_TOOLS = ["search_evidence", "read_evidence", FINDING_TOOL, "structured_facts", "aggregate_facts",
              "metadata_table", "citations", "evidence_for_claim"]


def intent(name: str) -> tuple[str, str]:
    evidence_type = {
        "search_evidence": "ranked_passage", "read_evidence": "passage",
        "structured_facts": "canonical_metadata", "evidence_for_claim": "claim_evidence",
        "facts": "canonical_metadata", "citations": "citation_fact", "search": "passage",
        "passages": "passage", "tables": "table_cell", "rank_evidence": "ranked_passage",
        "prior_triples": "claim_subgraph", "related": "related_entity",
        "record_finding": "research_note",
    }.get(name, "corpus_evidence")
    return (f"Collect {evidence_type.replace('_', ' ')} relevant to the active research question.", evidence_type)


def serialise(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, (list, tuple)):
        return [serialise(item) for item in value]
    if isinstance(value, dict):
        return {str(key): serialise(item) for key, item in value.items()}
    return value


def tool_payload(result: Any) -> Any:
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


def result_count(result: Any) -> int | None:
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


class TurnLimitReached(RuntimeError):
    """The agent used every turn it is allowed without a final decision (ends the loop like the deadline)."""


def _ids_of_claims(grounded: dict[str, Any]) -> set[str]:
    return {str(claim.get("id")) for claim in grounded.get("preliminary_claims", [])
            if isinstance(claim, dict) and claim.get("id") is not None}


class ToolLoop:
    """One agent conversation. ``client`` is the ``MCPToolClient`` (sink, agent name, call ids, notes)."""

    def __init__(self, client: Any, model: Any, session: Any, question: str,
                 grounded_context: dict[str, Any] | None) -> None:
        self.client, self.model, self.session, self.question = client, model, session, question
        # A local session (the evidence document's tools, doc_tools.py) brings its own tools and prompts.
        self.local = bool(getattr(session, "local", False))
        self.allowed = tuple(getattr(session, "allowed_tools", CURATED_TOOLS))
        self.config = settings()
        grounded = grounded_context or {}
        self.ledger = ResearchLedger(question=question)
        self.ledger.notes = client.notes
        self.conversation: Conversation | None = None
        self.grounded = grounded_context or None
        self.claim_ids = _ids_of_claims(grounded)
        self.shown_ids = _handles(grounded) | self.claim_ids
        self.needs_tools = grounded.get("answer_source") != "verified_facts"
        # Until the agent has called a corpus tool, ask for a tool call
        # outright ("required"): a non-thinking model otherwise often
        # answers from its own knowledge first and needs a rejected turn.
        self.first_choice = self.config.agent_first_tool_choice
        # One bounded self-correction: a decision that fails validation is
        # sent back once with the exact problems, instead of degrading the run.
        self.reflections_left = self.config.agent_reflections
        self.window = 0
        self.reply_tokens = 0
        self.retried_for_length = False
        self.budget_note_sent = False
        self.last_note_sent = False
        self.turn_number = 0
        budget = self.config.agent_max_tool_calls
        self.max_turns = budget + TURN_MARGIN if budget > 0 else 0
        self.tools: list[dict[str, Any]] = []
        self.core_count = 0

    # ─── Setup ───────────────────────────────────────────────────────────────

    async def _prepare(self) -> None:
        listed = await self.session.list_tools()
        tools = [{"type": "function", "function": {
            "name": item.name, "description": item.description or "", "parameters": item.inputSchema}}
            for item in listed.tools]
        tools = [tool for tool in tools if tool["function"]["name"] in self.allowed]
        tools.append(FINDING_TOOL_SPEC)
        tools.sort(key=lambda tool: (CORE_TOOLS.index(tool["function"]["name"])
                                     if tool["function"]["name"] in CORE_TOOLS else len(CORE_TOOLS)))
        self.tools = tools
        self.core_count = sum(1 for tool in tools if tool["function"]["name"] in CORE_TOOLS)
        loaded_skills = [] if self.local else relevant(self.question)
        for skill in loaded_skills:
            self.client.sink({
                "call_id": self.client.next_id("skill"), "agent": self.client.agent,
                "tool": "load_skill", "arguments": {"name": skill.name, "version": skill.version},
                "result": {"name": skill.name, "version": skill.version}, "result_count": 1,
                "duration_ms": 0.0, "status": "ok",
                "purpose": "Load a reusable research strategy for the active question.",
                "expected_evidence_type": "research_skill",
            })
        supervisor = render(
            "agent/evidence_doc_supervisor" if self.local else "agent/research_supervisor",
            skills_catalog=catalog(),
            loaded_skills="; ".join(f"{skill.name}@{skill.version}: {skill.instructions}" for skill in loaded_skills),
            skills=loaded_skills,
        )
        self.client.prompt_ref = supervisor.ref
        fixed: list[dict[str, Any]] = [
            {"role": "system", "content": supervisor.system},
            {"role": "user", "content": self.question},
        ]
        self.conversation = Conversation(fixed, grounded=self.grounded,
                                         keep_results=max(0, self.config.agent_keep_results),
                                         grounded_prompt="agent/evidence_doc_context" if self.local
                                         else "agent/grounded_context")
        self.window = discover_window(self.model)
        self.reply_tokens = min(self.config.agent_reply_tokens, max(512, self.window // 4))

    # ─── The loop ────────────────────────────────────────────────────────────

    async def run(self) -> str:
        await self._prepare()
        async with httpx.AsyncClient(timeout=self.model.timeout_s) as http:
            while True:
                try:
                    turn_timeout = bounded_timeout(float(self.model.timeout_s))
                except ModelFailure:
                    # Stopped or out of time: end the loop with what we have.
                    return ""
                budget_reached = self._budget_reached()
                last_turn = 0 < self.max_turns <= self.turn_number + 1
                offered = [] if last_turn else self._offered(budget_reached)
                await self._summarise_older(offered)
                self._compact(offered)
                if last_turn:
                    self._note_last_turn()
                else:
                    self._note_budget(offered, budget_reached)
                self._check_fits(offered)
                response = await self._send(http, offered, turn_timeout)
                if response is None:
                    return ""
                if response.status_code >= 400:
                    self._refit_or_raise(response)
                    continue
                message = self._receive(response)
                requested = message.get("tool_calls") or []
                if not requested:
                    final = str(message.get("content") or "")
                    if not last_turn and self._send_back(final):
                        continue
                    return final
                if last_turn:
                    self._stop_at_turn_limit(len(requested))
                await self._run_calls(message, requested, offered, budget_reached)

    def _budget_reached(self) -> bool:
        return 0 < self.config.agent_max_tool_calls <= self.ledger.tool_calls

    def _note_last_turn(self) -> None:
        """The last allowed turn: no tools are offered and the agent is told to decide now."""
        conversation = self.conversation
        if conversation.exchanges and not self.last_note_sent:
            self.last_note_sent = True
            conversation.exchanges[-1].append({"role": "user", "content": (
                f"This is your last turn (turn {self.max_turns} of {self.max_turns}): no more tool calls. "
                "Reply now with your final JSON decision.")})

    def _stop_at_turn_limit(self, requested: int) -> None:
        """The last turn asked for tools anyway: record it and end the loop like the phase deadline."""
        detail = (f"agent turn limit reached: {self.turn_number} turns (tool budget "
                  f"{self.config.agent_max_tool_calls} + {TURN_MARGIN}) without a final decision")
        self._checkpoint({"mode": "turn_limit"}, {
            "turns": self.turn_number, "max_turns": self.max_turns, "corpus_calls": self.ledger.tool_calls,
            "requested_on_last_turn": requested, "notes": len(self.ledger.notes),
        }, "Stop the agent loop at its turn limit.")
        emit_model_event("Agent turn limit", detail)
        raise TurnLimitReached(detail)

    def _offered(self, budget_reached: bool) -> list[dict[str, Any]]:
        """Tool budget reached: only the scratchpad is left, then the decision. Until a corpus tool
        has run, the first turn may force a tool call ("required"); the scratchpad is not offered
        then, so the forced call is a corpus call."""
        offered = [tool for tool in self.tools if tool["function"]["name"] == FINDING_TOOL] \
            if budget_reached else self.tools
        forcing = self.needs_tools and self.ledger.tool_calls == 0 and self.first_choice == "required"
        if forcing and any(tool["function"]["name"] != FINDING_TOOL for tool in offered):
            offered = [tool for tool in offered if tool["function"]["name"] != FINDING_TOOL]
        return offered

    def _note_budget(self, offered: list[dict[str, Any]], budget_reached: bool) -> None:
        conversation = self.conversation
        if budget_reached and not self.budget_note_sent and conversation.exchanges:
            self.budget_note_sent = True
            conversation.exchanges[-1].append({"role": "user", "content": (
                f"You have used the tool budget ({self.ledger.tool_calls} corpus calls). "
                + ("Record any finding you have not recorded yet, then reply"
                   if offered else "Reply") + " with your final JSON decision.")})

    def _check_fits(self, offered: list[dict[str, Any]]) -> None:
        needed = self.conversation.tokens(offered)
        if needed > self.limit():
            raise ModelFailure(
                f"agent context window ({self.window} tokens) is too small: the minimal "
                f"prompt with {len(offered)} core tools needs about {needed + self.reply_tokens} "
                "tokens; serve the agent model with a larger max_model_len or set "
                "HARNESS_AGENT_CONTEXT_TOKENS"
            )

    # ─── Context window ──────────────────────────────────────────────────────

    def limit(self) -> int:
        # Leave room for the reply and a margin for estimation error.
        return max(2_048, int((self.window - self.reply_tokens) * 0.9))

    def _checkpoint(self, arguments: dict[str, Any], result: dict[str, Any], purpose: str) -> None:
        self.client.sink({
            "call_id": self.client.next_id("context"), "agent": self.client.agent,
            "tool": "context_checkpoint", "arguments": arguments,
            "result": {**result, "evidence_ids": sorted(self.ledger.evidence_ids)},
            "result_count": len(self.ledger.evidence_ids), "duration_ms": 0.0, "status": "ok",
            "purpose": purpose, "expected_evidence_type": "research_ledger",
        })

    async def _summarise_older(self, sent_tools: list[dict[str, Any]]) -> None:
        """Near the limit, replace older exchanges with one model-written summary (Phase 3c).

        Only when the older exchanges (already masked to digests) hold enough tokens for the
        summary to save something; otherwise fitting handles the overflow without a model call.
        """
        conversation, model = self.conversation, self.model
        older = len(conversation.exchanges) - max(1, conversation.keep_results)
        if (not self.config.agent_summary_compaction or older < 2
                or conversation.tokens(sent_tools) < SUMMARISE_AT * self.limit()
                or conversation.exchange_tokens(older) < max(SUMMARY_MIN_SAVING, 0.1 * self.limit())
                or not callable(getattr(model, "complete_json", None))):
            return
        archived = []
        for exchange, digests in zip(conversation.exchanges[:older], conversation.digests[:older], strict=False):
            requested = [{"tool": call["function"]["name"], "arguments": call["function"].get("arguments")}
                         for call in exchange[0].get("tool_calls") or []]
            archived.append({"requested": requested, "returned": digests})
        prompt = render("agent/compaction_summary",
                        {"earlier_summary": conversation.summary, "archived": archived,
                         "findings": self.ledger.notes, "question": self.question})
        before = conversation.tokens(sent_tools)
        try:
            with using_prompt(prompt.ref):
                reply = await asyncio.to_thread(model.complete_json, prompt.system, prompt.user)
            summary = str(reply.get("summary") or "").strip()
            open_items = [str(item) for item in reply.get("open") or [] if str(item).strip()]
            if open_items:
                summary += " Still open: " + "; ".join(open_items[:5])
        except Exception as exc:  # noqa: BLE001 - the deterministic ledger still replaces them
            emit_model_event("Context summary failed", f"{type(exc).__name__}: {exc}"[:200])
            summary = ""
        conversation.archive_oldest(older, summary[:1500] or None, self.ledger.summary())
        self._checkpoint({"mode": "summary"}, {
            "window": self.window, "before_tokens": before, "after_tokens": conversation.tokens(sent_tools),
            "summarised_exchanges": older, "summary": summary[:1500],
        }, "Summarise older research to stay within the context window.")

    def _compact(self, sent_tools: list[dict[str, Any]]) -> None:
        conversation = self.conversation
        before = conversation.tokens(sent_tools)
        if conversation.fit(self.limit(), sent_tools, self.ledger.summary(), core_tools=self.core_count):
            self._checkpoint({}, {
                "window": self.window, "before_tokens": before, "after_tokens": conversation.tokens(sent_tools),
                "tools_offered": len(sent_tools), "tools_available": len(self.tools),
                "archived_exchanges": conversation.dropped_exchanges,
                "masked_results": conversation.masked_results, "token_scale": conversation.scale,
            }, "Keep the conversation within the model context window.")

    # ─── One model turn ──────────────────────────────────────────────────────

    @property
    def alias(self) -> str:
        return getattr(self.model, "model", "agent")

    async def _send(self, http: httpx.AsyncClient, offered: list[dict[str, Any]],
                    turn_timeout: float) -> httpx.Response | None:
        """Post one turn; None when Stop or the run deadline abandoned it."""
        self.turn_number += 1
        self.turn_started = time.monotonic()
        emit_model_event("Model call started",
                         f"{self.alias} · agent tool loop turn {self.turn_number} · {len(offered)} tools")
        body: dict[str, Any] = {
            "model": getattr(self.model, "model", ""),
            "temperature": 0,
            "max_tokens": self.reply_tokens,
            "messages": self.conversation.messages(),
        }
        if offered:
            body["tools"] = offered
            # Force only when a corpus tool is on offer: a forced scratchpad call never
            # counts as research, so every later turn would be forced too.
            body["tool_choice"] = self.first_choice if (
                self.needs_tools and self.ledger.tool_calls == 0
                and any(tool["function"]["name"] != FINDING_TOOL for tool in offered)) else "auto"
        self.sent_estimate = self.conversation.raw_tokens(offered)
        turn = asyncio.ensure_future(http.post(
            f"{self.model.base_url}/chat/completions",
            timeout=turn_timeout,
            headers={"Authorization": f"Bearer {self.model.api_key}"},
            json=body,
        ))
        # Wait in short intervals so Stop or the run deadline can
        # abandon a slow model turn immediately.
        while not turn.done():
            await asyncio.wait({turn}, timeout=0.25)
            if not turn.done() and call_interrupted():
                turn.cancel()
                return None
        try:
            return turn.result()
        except httpx.TimeoutException:
            emit_model_event("Model call failed", f"{self.alias} · agent tool loop turn "
                             f"{self.turn_number} · timed out after {time.monotonic() - self.turn_started:.0f}s")
            raise

    def _refit_or_raise(self, response: httpx.Response) -> None:
        """A context-length error once teaches the real window (the turn is retried); else fail."""
        body = response.text[:600]
        emit_model_event("Model call failed", f"{self.alias} · agent tool loop turn "
                         f"{self.turn_number} · HTTP {response.status_code}: {body[:200]}")
        if is_context_error(body) and not self.retried_for_length:
            self.retried_for_length = True
            learned = window_from_error(body)
            self.window = min(self.window, learned) if learned else int(self.window * 0.6)
            self.reply_tokens = min(self.config.agent_reply_tokens, max(512, self.window // 4))
            return
        raise ModelFailure(f"model returned HTTP {response.status_code}: {body}")

    def _receive(self, response: httpx.Response) -> dict[str, Any]:
        payload = response.json()
        message = payload["choices"][0]["message"]
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        self.conversation.calibrate(usage.get("prompt_tokens"), self.sent_estimate)
        requested = message.get("tool_calls") or []
        emit_model_event(
            "Model call finished",
            f"{self.alias} · agent tool loop turn {self.turn_number} · "
            f"{time.monotonic() - self.turn_started:.1f}s · "
            + (f"{len(requested)} tool calls" if requested else "final answer")
            + (f" · prompt {self.client.prompt_ref}" if self.turn_number == 1 and self.client.prompt_ref else ""),
        )
        return message

    def _send_back(self, final: str) -> bool:
        """Whether the decision has problems and a reflection is left (it is then sent back)."""
        problems = decision_problems(
            final, observed=self.shown_ids | self.ledger.evidence_ids, claim_ids=self.claim_ids,
            corpus_calls=self.ledger.tool_calls, needs_tools=self.needs_tools,
        )
        if not problems or self.reflections_left <= 0:
            return False
        self.reflections_left -= 1
        emit_model_event("Agent decision sent back", "; ".join(problems)[:400])
        self.conversation.add_exchange(
            {"role": "assistant", "content": final},
            [{"role": "user", "content": text("agent/decision_rejected", problems=problems)}],
        )
        return True

    # ─── Tool calls ──────────────────────────────────────────────────────────

    async def _run_calls(self, message: dict[str, Any], requested: list[dict[str, Any]],
                         offered: list[dict[str, Any]], budget_reached: bool) -> None:
        offered_names = {tool["function"]["name"] for tool in offered}
        tool_messages: list[dict[str, Any]] = []
        digests: list[str] = []
        for call in requested:
            # The budget is checked per call: parallel calls of one turn stop at it too.
            budget_reached = budget_reached or self._budget_reached()
            tool_message, digest = await self._run_call(call, offered_names, budget_reached)
            tool_messages.append(tool_message)
            digests.append(digest)
        # An exchange is only added whole: the assistant request
        # together with one result per requested call.
        assistant = {"role": "assistant", "content": message.get("content"), "tool_calls": requested}
        self.conversation.add_exchange(assistant, tool_messages, digests)

    async def _run_call(self, call: dict[str, Any], offered_names: set[str],
                        budget_reached: bool) -> tuple[dict[str, Any], str]:
        name = call["function"]["name"]
        started = time.monotonic()
        arguments, result, status, error = _parse_arguments(call)
        if error is None:
            repeat = self.ledger.repeat_of(name, arguments) if name != FINDING_TOOL else None
            if name not in offered_names:
                # Not offered this turn (budget reached, or not in the curated set): refuse
                # without calling MCP, and do not count it as research.
                result = {"error": f"tool {name} is not available now"
                          + ("; the tool budget is used up, reply with your final JSON decision"
                             if budget_reached else "")}
                status, error = "unavailable", result["error"]
            elif name == FINDING_TOOL:
                result, status, error = self._record_finding(arguments)
            elif budget_reached:
                # Offered at the start of the turn, but earlier calls of this turn used the
                # budget up: refuse it (not executed, not counted).
                result = {"error": f"unavailable: tool budget reached ({self.ledger.tool_calls} corpus "
                                   "calls); record findings, then reply with your final JSON decision"}
                status, error = "unavailable", result["error"]
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
                result, status, error = await self._call_corpus(name, arguments)
        record = self._report(name, arguments, result, status, error, started)
        ledger_digest = None
        if status not in {"repeat", "unavailable"} and name != FINDING_TOOL:
            self.ledger.record(name, result, record["purpose"], arguments=arguments, call_id=record["call_id"])
            ledger_digest = self.ledger.digests[-1]
        return {
            "role": "tool",
            "tool_call_id": call["id"],
            # The full result is kept in tool_calls.jsonl; the
            # model sees a page bounded by the window.
            "content": cap_tool_result(json.dumps(result, ensure_ascii=False, default=str), self.limit() // 4),
        }, result_digest(name, result, ledger_digest)

    def _record_finding(self, arguments: dict[str, Any]) -> tuple[Any, str, str | None]:
        observed_now = self.shown_ids | self.ledger.evidence_ids
        kept, problems = validate_findings([arguments], observed_now)
        if not kept:
            message = "; ".join(problems) or "invalid finding"
            return {"error": message}, "error", message
        note = self.ledger.add_note(kept[0]["statement"], kept[0]["evidence_ids"])
        dropped = [value for value in id_list(arguments.get("evidence_ids")) if value not in observed_now]
        result: dict[str, Any] = {"recorded": note["id"], "notes_so_far": len(self.ledger.notes)}
        if dropped:
            # Not "*_ids": evidence_ids() must not mistake these for evidence.
            result["removed"] = dropped
            result["note"] = "ids never returned by a tool were removed"
        return result, "ok", None

    async def _call_corpus(self, name: str, arguments: dict[str, Any]) -> tuple[Any, str, str | None]:
        status, error = "ok", None
        try:
            raw = await self.session.call_tool(name, arguments)
            result = tool_payload(serialise(raw))
        except Exception as exc:  # noqa: BLE001
            result = {"error": str(exc)}
            status, error = "error", str(exc)
        hint = self.ledger.loop_hint(name, arguments)
        if hint and isinstance(result, dict):
            result = {**result, "hint": hint}
            emit_model_event("Tool-loop hint", f"{name} called for several papers; aggregate tools suggested")
        return result, status, error

    def _report(self, name: str, arguments: dict[str, Any], result: Any, status: str, error: str | None,
                started: float) -> dict[str, Any]:
        record = {
            "call_id": self.client.next_id("tool"),
            "agent": self.client.agent,
            "tool": name,
            "arguments": arguments,
            "result": result,
            "result_count": result_count(result),
            "duration_ms": round((time.monotonic() - started) * 1000, 1),
            "status": status,
        }
        record["purpose"], record["expected_evidence_type"] = intent(name)
        if error:
            record["error"] = error
        self.client.sink(record)
        # Live progress: which tool ran and what it returned.
        emit_model_event("Tool call", f"{name} · " + (
            f"error: {str(error)[:160]}" if error
            else "repeated call answered from the ledger" if status == "repeat"
            else "returned" if record["result_count"] is None
            else f"{record['result_count']} results"
        ) + f" · {record['duration_ms']:.0f} ms")
        return record


def _parse_arguments(call: dict[str, Any]) -> tuple[dict[str, Any], Any, str, str | None]:
    """(arguments, result, status, error). A malformed call is reported back to the model as that
    tool's result so it can correct itself."""
    try:
        arguments = json.loads(call["function"].get("arguments") or "{}")
        if not isinstance(arguments, dict):
            raise TypeError("tool arguments must be a JSON object")
    except (json.JSONDecodeError, TypeError) as exc:
        result = {"error": f"invalid tool arguments: {exc}"}
        return {"_raw": str(call["function"].get("arguments"))[:500]}, result, "error", result["error"]
    return arguments, None, "ok", None
