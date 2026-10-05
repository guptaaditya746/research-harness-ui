"""Row helpers shared by the corpus store and its tools: reading JSONL, paging, scope, unknown ids."""

from __future__ import annotations

import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


class UnknownIdentifier(ValueError):
    """An identifier the corpus does not know; the message says what is accepted."""


def paged(rows: list[dict], offset: int = 0, limit: int = 10) -> dict:
    if offset < 0 or limit < 1:
        raise ValueError("offset must be nonnegative and limit must be positive")
    return {
        "total": len(rows),
        "offset": offset,
        "limit": limit,
        "items": rows[offset : offset + limit],
    }


def in_scope(paper: dict, scope: dict | None) -> bool:
    scope = scope or {}
    ids = scope.get("paper_ids")
    if ids and paper["paper_id"] not in ids:
        return False
    years = scope.get("year_range")
    return not years or paper["year"] is not None and years[0] <= paper["year"] <= years[1]
