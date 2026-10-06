"""The agent's tools when the evidence document D is the run's input (docs/RUN_INPUT.md).

A stand-in for the MCP corpus session with the same two calls the tool loop makes (``list_tools``,
``call_tool``), answered in process over D's triples only: the agent cannot reach the corpus. Results
name each triple by ``triple_id`` and its passage by ``source_id``, so findings and the final decision can
cite them (the loop only accepts ids a tool returned).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from types import SimpleNamespace
from typing import Any

from harness_baseline.evidence_doc import DocTriple, EvidenceDoc
from harness_config import settings

DOC_TOOLS = ("doc_overview", "doc_search", "doc_triples", "doc_read")

_SPECS: dict[str, tuple[str, dict[str, Any]]] = {
    "doc_overview": (
        ("What the evidence document holds: how many triples, its interpretations, its relations and its "
         "papers with their triple counts. Call it first."), {"type": "object", "properties": {}}),
    "doc_search": (
        "Rank the document's triples by the words of a query (subject, relation, object and passage text).",
        {"type": "object", "properties": {
            "query": {"type": "string", "description": "Words to look for."},
            "limit": {"type": "integer", "description": "How many triples (default: one page)."}},
         "required": ["query"]}),
    "doc_triples": (
        ("List the document's triples, optionally only one relation, subject, paper or interpretation, a "
         "page at a time."),
        {"type": "object", "properties": {
            "predicate": {"type": "string"}, "subject": {"type": "string"}, "paper_id": {"type": "string"},
            "interpretation": {"type": "string", "description": "An interpretation id from doc_overview."},
            "offset": {"type": "integer"}, "limit": {"type": "integer"}}}),
    "doc_read": (
        "The full passage evidence of some triples, by triple_id.",
        {"type": "object", "properties": {"triple_ids": {"type": "array", "items": {"type": "string"}}},
         "required": ["triple_ids"]}),
}


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(text)).casefold())


def _row(triple: DocTriple, *, full: bool = False) -> dict[str, Any]:
    row = triple.row()
    row["triple_id"] = row.pop("id")
    row["interpretation"] = triple.reading
    if full:
        row["evidence"] = triple.evidence
    return row


class EvidenceDocSession:
    """``list_tools`` / ``call_tool`` over D (async, like the MCP ``ClientSession``)."""

    allowed_tools = DOC_TOOLS
    local = True

    def __init__(self, doc: EvidenceDoc) -> None:
        self.doc = doc
        self.by_id = {triple.id: triple for triple in doc.triples}
        self.page = settings().evidence_doc_page
        self._text = [_words(f"{t.s} {t.p} {t.o} {t.evidence}") for t in doc.triples]
        frequency = Counter(word for words in self._text for word in set(words))
        self._idf = {word: math.log(1 + len(doc.triples) / count) for word, count in frequency.items()}

    async def list_tools(self) -> SimpleNamespace:
        return SimpleNamespace(tools=[SimpleNamespace(name=name, description=description, inputSchema=schema)
                                      for name, (description, schema) in _SPECS.items()])

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        handler = getattr(self, f"_{name}", None)
        if name not in DOC_TOOLS or handler is None:
            return {"error": f"unknown tool {name}; the tools are {', '.join(DOC_TOOLS)}"}
        return handler(**{key: value for key, value in arguments.items() if not key.startswith("_")})

    def _limit(self, value: object) -> int:
        try:
            return max(1, min(int(value), self.page * 4))
        except (TypeError, ValueError):
            return self.page

    def _doc_overview(self) -> dict[str, Any]:
        triples = self.doc.triples
        return {
            "triples": len(triples), "question_id": self.doc.question_id,
            "interpretations": [{**item, "triples": sum(t.reading == item["id"] for t in triples)}
                                for item in self.doc.interpretations],
            "relations": Counter(t.p for t in triples).most_common(40),
            "papers": [{"paper_id": paper, "triples": count}
                       for paper, count in Counter(t.paper_id or "?" for t in triples).most_common(60)],
            "page_size": self.page,
        }

    def _doc_search(self, query: str = "", limit: object = None) -> dict[str, Any]:
        wanted = set(_words(query))
        scored = [(sum(self._idf.get(word, 0.0) for word in wanted & set(words)), index)
                  for index, words in enumerate(self._text)]
        ranked = [index for score, index in sorted(scored, key=lambda item: (-item[0], item[1])) if score > 0]
        items = [_row(self.doc.triples[index]) for index in ranked[: self._limit(limit)]]
        return {"total": len(ranked), "items": items}

    def _doc_triples(self, predicate: str = "", subject: str = "", paper_id: str = "", interpretation: str = "",
                     offset: object = 0, limit: object = None) -> dict[str, Any]:
        chosen = [t for t in self.doc.triples
                  if (not predicate or t.p.casefold() == predicate.casefold())
                  and (not subject or subject.casefold() in t.s.casefold())
                  and (not paper_id or (t.paper_id or "") == paper_id)
                  and (not interpretation or t.reading == interpretation)]
        try:
            start = max(0, int(offset or 0))
        except (TypeError, ValueError):
            start = 0
        page = chosen[start:start + self._limit(limit)]
        return {"total": len(chosen), "offset": start, "items": [_row(t) for t in page],
                **({"next_offset": start + len(page)} if start + len(page) < len(chosen) else {})}

    def _doc_read(self, triple_ids: object = None) -> dict[str, Any]:
        ids = [str(item) for item in triple_ids] if isinstance(triple_ids, list) else [str(triple_ids or "")]
        found = [_row(self.by_id[item], full=True) for item in ids if item in self.by_id]
        missing = [item for item in ids if item not in self.by_id]
        return {"items": found, **({"unknown": missing} if missing else {})}


def chosen_triples(doc: EvidenceDoc, ids: set[str]) -> set[str]:
    """D's triple ids the agent named: triple ids directly, and every triple of a named source passage."""
    return {t.id for t in doc.triples if t.id in ids or t.source_id in ids}


__all__ = ["DOC_TOOLS", "EvidenceDocSession", "chosen_triples"]
