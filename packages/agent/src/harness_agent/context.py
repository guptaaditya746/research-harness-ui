"""Context-window management for the agent's tool conversation.

The agent conversation must always fit the model's real context window, and
it must stay a *valid* OpenAI-style conversation: an assistant message that
requests tools is followed by exactly one tool message per requested call.
Earlier compaction kept "the last four messages", which could start with a
tool message whose assistant request had been cut; OpenAI-compatible servers
reject that with 400 Bad Request.

Here the conversation is three fixed parts (system, question, grounded
context) plus complete *exchanges* (assistant tool request + its tool
results). Three mechanisms keep it inside the window, cheapest first:

1. **Observation masking**: tool results older than the last ``keep_results``
   exchanges are shown as a short digest (ids and first items), the request
   itself stays, so the conversation remains valid and the ids remain known.
2. **Summarising compaction** (in the tool loop): near the limit, older
   exchanges are replaced by one model-written summary plus the research
   ledger (digests and the agent's findings).
3. **Fitting**: drops whole exchanges oldest-first, then specialist tools, then
   grounded-context tails, then truncates the last results with a note.

Token counts are estimated (~3 characters per token) and calibrated against
the ``usage.prompt_tokens`` the model server reports for each turn.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import httpx
from harness_config import settings
from harness_extraction.model_client import CHARS_PER_TOKEN
from harness_prompts import text

DEFAULT_WINDOW = 32_768


def estimate_tokens(value: Any) -> int:
    """Conservative token estimate for JSON-heavy chat payloads (``CHARS_PER_TOKEN``, ~3 chars/token)."""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return int(len(text) / CHARS_PER_TOKEN) + 1


_WINDOW_PATTERNS = (
    re.compile(r"maximum context length is (\d+)", re.IGNORECASE),
    re.compile(r"max_model_len[^\d]{0,20}(\d+)", re.IGNORECASE),
    re.compile(r"context (?:window|length)[^\d]{0,40}(\d{4,7})", re.IGNORECASE),
)


def window_from_error(text: str) -> int | None:
    """Read the model's context window from a provider's length error, if present."""
    for pattern in _WINDOW_PATTERNS:
        match = pattern.search(text)
        if match:
            return int(match.group(1))
    return None


def is_context_error(text: str) -> bool:
    lowered = text.casefold()
    return any(marker in lowered for marker in (
        "context length", "context window", "maximum context", "max_model_len",
        "too many tokens", "prompt is too long", "input is too long",
    ))


def discover_window(model: Any) -> int:
    """The agent model's context window: configured, reported by LiteLLM, or a default."""
    configured = settings().agent_context_tokens
    if configured:
        return int(configured)
    base = str(getattr(model, "base_url", "")).rstrip("/")
    alias = str(getattr(model, "model", ""))
    if base and alias:
        root = base.removesuffix("/v1")
        headers = {"Authorization": f"Bearer {getattr(model, 'api_key', '')}"}
        for url in (f"{root}/model/info", f"{base}/model/info"):
            try:
                response = httpx.get(url, headers=headers, timeout=5)
                if response.status_code != 200:
                    continue
                for entry in response.json().get("data", []):
                    if entry.get("model_name") != alias:
                        continue
                    info = entry.get("model_info") or {}
                    for key in ("max_input_tokens", "max_model_len", "max_tokens"):
                        if isinstance(info.get(key), int) and info[key] > 0:
                            return int(info[key])
            except (httpx.HTTPError, ValueError):
                continue
    return DEFAULT_WINDOW


SHRINKABLE = ("preliminary_claims", "preliminary_triples", "papers", "concept_evidence")


@dataclass
class Conversation:
    """A tool conversation that can always be fitted into a token budget."""

    fixed: list[dict[str, Any]]
    grounded: dict[str, Any] | None = None
    exchanges: list[list[dict[str, Any]]] = field(default_factory=list)
    ledger_note: dict[str, Any] | None = None
    dropped_exchanges: int = 0
    shown: dict[str, int] = field(default_factory=dict)
    # Observation masking: results of exchanges older than this many are shown as digests (0 = off).
    keep_results: int = 0
    digests: list[list[str]] = field(default_factory=list)
    # Estimated tokens × scale ≈ the server's count (calibrated from usage.prompt_tokens).
    scale: float = 1.0
    summary: str | None = None
    masked_results: int = 0
    # The template that frames the grounded context (agent/evidence_doc_context when D is the input).
    grounded_prompt: str = "agent/grounded_context"

    def _grounded_message(self) -> dict[str, Any] | None:
        if not self.grounded:
            return None
        rendered = dict(self.grounded)
        omitted = {}
        for key in SHRINKABLE:
            items = rendered.get(key)
            if isinstance(items, list) and key in self.shown and self.shown[key] < len(items):
                omitted[key] = len(items) - self.shown[key]
                rendered[key] = items[: self.shown[key]]
        if omitted:
            rendered["omitted_for_context"] = {
                **omitted,
                "note": "Listed items come first in answer order; read the rest with the corpus "
                        "tools (structured_facts, search_evidence, read_evidence).",
            }
        return {"role": "user", "content": text(
            self.grounded_prompt, context_json=json.dumps(rendered, ensure_ascii=False))}

    def messages(self) -> list[dict[str, Any]]:
        head = list(self.fixed)
        grounded = self._grounded_message()
        if grounded is not None:
            head.append(grounded)
        if self.ledger_note is not None:
            head.append(self.ledger_note)
        body: list[dict[str, Any]] = []
        masked = 0
        for index in range(len(self.exchanges)):
            rendered, count = self._rendered_exchange(index)
            body.extend(rendered)
            masked += count
        self.masked_results = masked
        return head + body

    def _rendered_exchange(self, index: int) -> tuple[list[dict[str, Any]], int]:
        """One exchange as sent: older ones with their tool results replaced by digests."""
        exchange = self.exchanges[index]
        cutoff = len(self.exchanges) - self.keep_results if self.keep_results else 0
        if index >= cutoff:
            return list(exchange), 0
        results = iter(self.digests[index] if index < len(self.digests) else [])
        rendered, masked = [], 0
        for message in exchange:
            if message.get("role") == "tool":
                digest = next(results, None)
                if digest is not None:
                    message = {**message, "content": digest}
                    masked += 1
            rendered.append(message)
        return rendered, masked

    def exchange_tokens(self, count: int) -> int:
        """Calibrated tokens of the oldest ``count`` exchanges as sent (what archiving them would free)."""
        messages = [message for index in range(min(count, len(self.exchanges)))
                    for message in self._rendered_exchange(index)[0]]
        return int(estimate_tokens(messages) * self.scale) if messages else 0

    def _shrink_grounded(self) -> bool:
        """Halve the largest list in the grounded context; False when nothing is left to cut."""
        if not self.grounded:
            return False
        candidates = [
            (self.shown.get(key, len(self.grounded[key])), key) for key in SHRINKABLE
            if isinstance(self.grounded.get(key), list) and self.shown.get(key, len(self.grounded[key])) > 3
        ]
        if not candidates:
            return False
        count, key = max(candidates)
        self.shown[key] = max(3, count // 2)
        return True

    def raw_tokens(self, tools: list[dict[str, Any]]) -> int:
        """Uncalibrated estimate (what calibration compares with the server's count)."""
        return estimate_tokens(self.messages()) + estimate_tokens(tools)

    def tokens(self, tools: list[dict[str, Any]]) -> int:
        return int(self.raw_tokens(tools) * self.scale)

    def calibrate(self, prompt_tokens: object, estimated: int) -> None:
        """Move the scale towards the server's real count for the last request (bounded, smoothed)."""
        if not isinstance(prompt_tokens, int) or prompt_tokens <= 0 or estimated <= 0:
            return
        observed = min(2.5, max(0.5, prompt_tokens / estimated))
        self.scale = round(0.5 * self.scale + 0.5 * observed, 3)

    def add_exchange(self, assistant: dict[str, Any], tool_messages: list[dict[str, Any]],
                     digests: list[str] | None = None) -> None:
        self.exchanges.append([assistant, *tool_messages])
        self.digests.append(list(digests or []))

    def archive_oldest(self, count: int, summary: str | None, ledger_summary: str) -> None:
        """Replace the oldest ``count`` exchanges with a summary and the research ledger."""
        count = min(count, max(0, len(self.exchanges) - 1))
        if count <= 0:
            return
        del self.exchanges[:count]
        del self.digests[:count]
        self.dropped_exchanges += count
        if summary:
            self.summary = summary
        self._set_ledger_note(ledger_summary)

    def _set_ledger_note(self, ledger_summary: str) -> None:
        summary = f"Summary of the archived research: {self.summary}\n" if self.summary else ""
        self.ledger_note = {
            "role": "user",
            "content": (
                f"{summary}Research ledger ({self.dropped_exchanges} earlier tool exchanges archived; "
                "evidence IDs below remain valid and can be re-read with tools): " + ledger_summary
            ),
        }

    def fit(
        self, limit: int, tools: list[dict[str, Any]], ledger_summary: str, core_tools: int = 0,
    ) -> bool:
        """Shrink until the conversation fits ``limit`` tokens; True if anything changed.

        Order of sacrifice: archived tool exchanges, then specialist tools
        beyond the first ``core_tools`` (``tools`` is priority-ordered and is
        trimmed in place), then the tail of the grounded lists, then the
        length of the last tool results.
        """
        changed = False
        while self.tokens(tools) > limit and len(self.exchanges) > 1:
            self.exchanges.pop(0)
            if self.digests:
                self.digests.pop(0)
            self.dropped_exchanges += 1
            self._set_ledger_note(ledger_summary)
            changed = True
        while self.tokens(tools) > limit and len(tools) > core_tools:
            tools.pop()
            changed = True
        while self.tokens(tools) > limit and self._shrink_grounded():
            changed = True
        if self.tokens(tools) > limit and self.exchanges:
            # One exchange left and still too large: shorten its tool results.
            overflow = self.tokens(tools) - limit
            results = [m for m in self.exchanges[-1] if m.get("role") == "tool"]
            if results:
                cut_chars = int(overflow * CHARS_PER_TOKEN / self.scale) // len(results) + 200
                for message in results:
                    content = str(message.get("content") or "")
                    keep = max(400, len(content) - cut_chars)
                    if keep < len(content):
                        message["content"] = content[:keep] + (
                            ' … [truncated to fit the context window; request a narrower page '
                            'with offset/limit or read specific IDs]'
                        )
                        changed = True
        return changed


def result_digest(tool: str, result: Any, ledger_digest: dict[str, Any] | None = None) -> str:
    """What an older tool result is replaced with: its ids and first items, not its full text."""
    body: dict[str, Any] = {"earlier_result_of": tool}
    if ledger_digest:
        body.update({key: value for key, value in ledger_digest.items() if key not in {"tool", "args"}})
    else:
        text = json.dumps(result, ensure_ascii=False, default=str)
        body["excerpt"] = text[:300] + ("…" if len(text) > 300 else "")
    body["note"] = "full result removed to save context; read ids again with read_evidence if needed"
    return json.dumps(body, ensure_ascii=False)


def cap_tool_result(content: str, limit_tokens: int) -> str:
    """Bound one tool result to ``limit_tokens`` (the tool loop passes a quarter of the usable window),
    converted to characters with the same ``CHARS_PER_TOKEN`` as ``estimate_tokens``; at least 2,000
    characters. Several results together are kept inside the window by ``Conversation.fit``."""
    max_chars = max(2_000, int(limit_tokens * CHARS_PER_TOKEN))
    if len(content) <= max_chars:
        return content
    return content[:max_chars] + (
        ' … [result truncated; request a narrower page with offset/limit or read specific IDs]'
    )


__all__ = [
    "Conversation", "cap_tool_result", "discover_window", "estimate_tokens",
    "is_context_error", "result_digest", "window_from_error",
]
