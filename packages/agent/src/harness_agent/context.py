"""Context-window management for the agent's tool conversation.

The agent conversation must always fit the model's real context window, and
it must stay a *valid* OpenAI-style conversation: an assistant message that
requests tools is followed by exactly one tool message per requested call.
Earlier compaction kept "the last four messages", which could start with a
tool message whose assistant request had been cut; OpenAI-compatible servers
reject that with 400 Bad Request.

Here the conversation is three fixed parts (system, question, grounded
context) plus complete *exchanges* (assistant tool request + its tool
results). Fitting drops whole exchanges oldest-first, replacing them with the
research ledger, and truncates oversized tool results with an explicit note.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import httpx
from harness_config import settings
from harness_prompts import text

DEFAULT_WINDOW = 32_768


def estimate_tokens(value: Any) -> int:
    """Conservative token estimate for JSON-heavy chat payloads (~3 chars/token)."""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return len(text) // 3 + 1


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


SHRINKABLE = ("preliminary_claims", "preliminary_triples", "papers")


@dataclass
class Conversation:
    """A tool conversation that can always be fitted into a token budget."""

    fixed: list[dict[str, Any]]
    grounded: dict[str, Any] | None = None
    exchanges: list[list[dict[str, Any]]] = field(default_factory=list)
    ledger_note: dict[str, Any] | None = None
    dropped_exchanges: int = 0
    shown: dict[str, int] = field(default_factory=dict)

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
            "agent/grounded_context", context_json=json.dumps(rendered, ensure_ascii=False))}

    def messages(self) -> list[dict[str, Any]]:
        head = list(self.fixed)
        grounded = self._grounded_message()
        if grounded is not None:
            head.append(grounded)
        if self.ledger_note is not None:
            head.append(self.ledger_note)
        return head + [message for exchange in self.exchanges for message in exchange]

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

    def tokens(self, tools: list[dict[str, Any]]) -> int:
        return estimate_tokens(self.messages()) + estimate_tokens(tools)

    def add_exchange(self, assistant: dict[str, Any], tool_messages: list[dict[str, Any]]) -> None:
        self.exchanges.append([assistant, *tool_messages])

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
            self.dropped_exchanges += 1
            self.ledger_note = {
                "role": "user",
                "content": (
                    f"Research ledger ({self.dropped_exchanges} earlier tool exchanges archived; "
                    "evidence IDs below remain valid and can be re-read with tools): " + ledger_summary
                ),
            }
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
                cut_chars = (overflow * 3) // len(results) + 200
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


def cap_tool_result(content: str, limit_tokens: int) -> str:
    """Bound one tool result so a single large page cannot fill the window."""
    max_chars = max(2_000, limit_tokens * 3 // 4)
    if len(content) <= max_chars:
        return content
    return content[:max_chars] + (
        ' … [result truncated; request a narrower page with offset/limit or read specific IDs]'
    )


__all__ = [
    "Conversation", "cap_tool_result", "discover_window", "estimate_tokens",
    "is_context_error", "window_from_error",
]
