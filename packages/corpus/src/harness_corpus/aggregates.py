"""Deterministic corpus-wide aggregates (``CorpusStore.aggregate``), each with the ids it used.

One function per operation returns ``(items, unit)``; ``aggregate`` sorts, filters and adds coverage.
"""

from __future__ import annotations

import re
from typing import Any

from .rows import UnknownIdentifier


def _ids(rows: list[dict]) -> dict[str, list[str]]:
    return {"fact_ids": sorted(str(row.get("fact_id") or row["source_id"]) for row in rows),
            "source_ids": sorted({str(row["source_id"]) for row in rows})}


def _by_author(facts: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in facts:
        if row.get("predicate") == "author":
            grouped.setdefault(str(row["object"]), []).append(row)
    return grouped


def count_papers_by_author(store: Any, facts: list[dict], predicate: str | None) -> tuple[list[dict], str]:
    items = []
    for author, rows in _by_author(facts).items():
        works = store.distinct_works(rows)
        items.append({"author": author, "count": len(works),
                      "paper_ids": sorted(str(row["subject"]) for row in works), **_ids(works)})
    return items, "distinct papers"


def count_by_value(store: Any, facts: list[dict], predicate: str | None) -> tuple[list[dict], str]:
    key = store.check_predicate(predicate or "")
    grouped: dict[str, list[dict]] = {}
    for row in facts:
        if row.get("predicate") == key:
            grouped.setdefault(str(row["object"]), []).append(row)
    items = [{"value": value, "count": len({str(row["subject"]) for row in rows}),
              "paper_ids": sorted({str(row["subject"]) for row in rows}), **_ids(rows)}
             for value, rows in grouped.items()]
    return items, "papers"


def _extreme(operation: str) -> Any:
    def compute(store: Any, facts: list[dict], predicate: str | None) -> tuple[list[dict], str]:
        key = store.check_predicate(predicate or "")
        numeric = [row for row in facts if row.get("predicate") == key
                   and re.fullmatch(r"-?\d+(?:\.\d+)?", str(row["object"]))]
        if not numeric:
            raise UnknownIdentifier(f"predicate {predicate!r} has no numeric values; try 'year'.")
        best = (min if operation == "min" else max)(float(row["object"]) for row in numeric)
        rows = [row for row in numeric if float(row["object"]) == best]
        return [{"value": rows[0]["object"], "count": len(rows),
                 "paper_ids": sorted(str(row["subject"]) for row in rows), **_ids(rows)}], f"{operation} {key}"
    return compute


def _incoming(store: Any) -> dict[str, list[dict]]:
    incoming: dict[str, list[dict]] = {}
    for reference in store.references:
        target = reference.get("resolved_paper_id")
        if target and target in store.by_paper:
            incoming.setdefault(str(target), []).append(reference)
    return incoming


def count_incoming_citations(store: Any, facts: list[dict], predicate: str | None) -> tuple[list[dict], str]:
    items = [{"paper_id": paper, "title": store.paper_title(paper),
              "count": len({str(ref["paper_id"]) for ref in refs}),
              "citing_paper_ids": sorted({str(ref["paper_id"]) for ref in refs}),
              "reference_ids": sorted(str(ref["reference_id"]) for ref in refs)}
             for paper, refs in _incoming(store).items()]
    return items, "citing papers"


def count_citations_by_author(store: Any, facts: list[dict], predicate: str | None) -> tuple[list[dict], str]:
    incoming = _incoming(store)
    items = []
    for author, rows in _by_author(facts).items():
        works = store.distinct_works(rows)
        refs = [ref for row in works for ref in incoming.get(str(row["subject"]), [])]
        items.append({"author": author, "count": len({str(ref["paper_id"]) for ref in refs}),
                      "paper_ids": sorted(str(row["subject"]) for row in works),
                      "reference_ids": sorted(str(ref["reference_id"]) for ref in refs),
                      "fact_ids": _ids(works)["fact_ids"]})
    return items, "citing papers (in-corpus) across the author's distinct papers"


OPERATIONS = {
    "count_papers_by_author": count_papers_by_author,
    "count_by_value": count_by_value,
    "min": _extreme("min"),
    "max": _extreme("max"),
    "count_incoming_citations": count_incoming_citations,
    "count_citations_by_author": count_citations_by_author,
}


def aggregate(store: Any, operation: str, predicate: str | None = None, top_k: int | None = None,
              min_count: int | None = None) -> dict:
    compute = OPERATIONS.get(operation)
    if compute is None:
        raise UnknownIdentifier(
            f"unknown operation {operation!r}. Use count_papers_by_author, count_by_value, min, max, "
            "count_incoming_citations or count_citations_by_author."
        )
    facts = [row for row in store.fact_rows if str(row.get("subject")) in store.by_paper]
    items, unit = compute(store, facts, predicate)
    items.sort(key=lambda item: (-int(item["count"]), str(item.get("author") or item.get("value") or item.get("paper_id"))))
    if min_count is not None:
        items = [item for item in items if item["count"] >= min_count]
    if top_k is not None and operation not in {"min", "max"} and items and top_k >= 1:
        cutoff = items[min(top_k, len(items)) - 1]["count"]
        items = [item for item in items if item["count"] >= cutoff]   # keep ties
    return {"operation": operation, "unit": unit, "computed": True, "total": len(items), "items": items,
            "coverage": {"scope": "all_indexed_papers", "exhaustive": True,
                         "papers_total": len(store.by_paper), "papers_examined": len(store.by_paper)}}
