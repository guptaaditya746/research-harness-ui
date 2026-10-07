"""Domain state retained when an agent conversation is compacted.

The ledger is the agent's memory across context compaction: instead of the
raw tool exchanges (which are dropped when the window fills) it keeps a
deterministic digest of every tool call — what was asked and the first items
returned, with their ids — so nothing found so far is lost when the
conversation is shortened. It also watches the tool loop for repeated
identical calls and per-paper loops that an aggregate tool answers at once.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

ID_KEYS = ("triple_id", "passage_id", "fact_id", "reference_id", "paper_id", "claim_id", "source_id")
BRIEF_CHARS = 120       # the id and identifying fields of one digest line
LIST_IDS_SHOWN = 5      # ids kept from each *_ids list of a row
BRIEF_MAX_CHARS = 320   # one digest line with its id lists
# Tools that answer for one paper at a time; a long run of these on different
# papers is what aggregate_facts / metadata_table replace.
PER_PAPER_TOOLS = {"structured_facts", "facts", "citations", "read_evidence", "evidence_for_claim"}
LOOP_HINT_AFTER = 4
DIGEST_ITEMS = 6
DIGESTS_KEPT = 12


def _handles(value: Any) -> set[str]:
    if isinstance(value, dict):
        found = {
            str(item)
            for key, item in value.items()
            if key.endswith("_id") and isinstance(item, (str, int))
        }
        for key, item in value.items():
            if key.endswith("_ids") and isinstance(item, list):
                found.update(str(entry) for entry in item if isinstance(entry, (str, int)))
        for item in value.values():
            found.update(_handles(item))
        return found
    if isinstance(value, list):
        return set().union(*(_handles(item) for item in value)) if value else set()
    return set()


def _id_lists(item: dict[str, Any]) -> str:
    """The row's ``*_ids`` lists, each cut to ``LIST_IDS_SHOWN`` ids with a "+k more" marker."""
    parts = []
    for key, value in item.items():
        if not key.endswith("_ids") or not isinstance(value, list):
            continue
        ids = [str(entry) for entry in value if isinstance(entry, (str, int))]
        if ids:
            more = len(ids) - LIST_IDS_SHOWN
            parts.append(f"{key}: {', '.join(ids[:LIST_IDS_SHOWN])}" + (f" +{more} more" if more > 0 else ""))
    return " | ".join(parts)


def _brief(item: Any) -> str:
    """One bounded line per returned item: its id, the fields that identify it and its id lists.

    A triple (the evidence document's tools) is named by its ``triple_id`` with the passage it comes
    from (``source_id``); aggregate rows keep the first ids of each ``*_ids`` list.
    """
    if not isinstance(item, dict):
        return str(item)[:100]
    handle_key = next((key for key in ID_KEYS if item.get(key) not in (None, "")), None)
    handle = str(item[handle_key]) if handle_key else None
    shown = {handle_key}
    if handle_key == "triple_id" and item.get("source_id") not in (None, ""):
        handle = f"{handle} (source {item['source_id']})"
        shown.add("source_id")
    if {"subject", "predicate", "object"} <= item.keys():
        body = f"{item['subject']} {item['predicate']} {item['object']}"
    else:
        body = " · ".join(
            str(value) for key, value in item.items()
            if key not in shown and not key.endswith("_ids") and isinstance(value, (str, int, float))
            and str(value).strip()
        )
    line = (f"{handle}: {body}" if handle else body)[:BRIEF_CHARS]
    lists = _id_lists(item)
    return f"{line} | {lists}"[:BRIEF_MAX_CHARS] if lists else line


def signature(tool: str, arguments: dict[str, Any] | None) -> str:
    return json.dumps({"tool": tool, "args": arguments or {}}, sort_keys=True, default=str)


@dataclass
class ResearchLedger:
    question: str
    interpretation: str | None = None
    findings: list[str] = field(default_factory=list)
    digests: list[dict[str, Any]] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    evidence_ids: set[str] = field(default_factory=set)
    tool_calls: int = 0
    # Loop guards.
    seen_calls: dict[str, str] = field(default_factory=dict)
    cached_results: dict[str, Any] = field(default_factory=dict)
    per_paper_calls: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    hinted: bool = False
    repeats: int = 0
    # The agent's own notes (record_finding): statements with the evidence ids behind them.
    notes: list[dict[str, Any]] = field(default_factory=list)

    def add_note(self, statement: str, evidence_ids: list[str]) -> dict[str, Any]:
        note = {"id": f"F{len(self.notes) + 1}", "statement": statement.strip()[:400],
                "evidence_ids": list(dict.fromkeys(evidence_ids))[:8]}
        self.notes.append(note)
        return note

    def record(self, tool: str, result: Any, purpose: str, *,
               arguments: dict[str, Any] | None = None, call_id: str | None = None) -> None:
        self.tool_calls += 1
        self.evidence_ids.update(_handles(result))
        items = result.get("items") if isinstance(result, dict) and isinstance(result.get("items"), list) else None
        count = len(items) if items is not None else None
        self.findings.append(f"{tool}: {count if count is not None else 'completed'} candidates ({purpose})")
        digest: dict[str, Any] = {"tool": tool}
        if arguments:
            digest["args"] = {
                key: value[:80] if isinstance(value, str)
                else value if isinstance(value, (int, float, bool)) else str(value)[:80]
                for key, value in arguments.items()
            }
        if isinstance(result, dict):
            if result.get("error"):
                digest["error"] = str(result["error"])[:160]
            if isinstance(result.get("total"), int):
                digest["total"] = result["total"]
        if items is not None:
            digest["returned"] = len(items)
            digest["items"] = [_brief(item) for item in items[:DIGEST_ITEMS]]
        self.digests.append(digest)
        key = signature(tool, arguments)
        if call_id and key not in self.seen_calls:
            self.seen_calls[key] = call_id
        self.cached_results[key] = result

    def repeat_of(self, tool: str, arguments: dict[str, Any] | None) -> tuple[str, Any] | None:
        """The earlier call id and result when this exact call was already made."""
        key = signature(tool, arguments)
        if key in self.seen_calls:
            self.repeats += 1
            return self.seen_calls[key], self.cached_results.get(key)
        return None

    def loop_hint(self, tool: str, arguments: dict[str, Any] | None) -> str | None:
        """Once, after several per-paper calls of one tool, point at the aggregate tools."""
        if tool not in PER_PAPER_TOOLS or not arguments:
            return None
        paper = arguments.get("paper_id")
        if paper:
            self.per_paper_calls[tool].add(str(paper))
        distinct = len(self.per_paper_calls[tool])
        if self.hinted or distinct < LOOP_HINT_AFTER:
            return None
        self.hinted = True
        return (
            f"You have called {tool} for {distinct} different papers. For counts, rankings, "
            "earliest/latest or 'which papers/authors' across the corpus call aggregate_facts once "
            "(it returns every match and cites its fact ids); for the metadata of many papers call "
            "metadata_table once. Continue only if you need one specific paper."
        )

    def summary(self) -> str:
        """What replaces archived exchanges: the digest of what each tool call found."""
        return json.dumps({
            "question": self.question,
            "tool_calls_so_far": self.tool_calls,
            "your_findings": self.notes,
            "found": self.digests[-DIGESTS_KEPT:],
            "unresolved": self.unresolved,
            "evidence_ids": sorted(self.evidence_ids),
        }, ensure_ascii=False)


__all__ = ["LOOP_HINT_AFTER", "PER_PAPER_TOOLS", "ResearchLedger", "signature"]
