"""Registry for evidence-backed query-local vocabulary.

The registry deliberately stores provisional relations separately from the
seed ontology.  A relation is reusable across runs, but cannot silently become
approved vocabulary.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path


class RelationRegistry:
    def __init__(self, path: Path):
        self.path = path
        with self._connect() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS relation_registry (
                id TEXT PRIMARY KEY, label TEXT NOT NULL, normalized_label TEXT NOT NULL UNIQUE,
                definition TEXT NOT NULL, domain_name TEXT, range_name TEXT,
                status TEXT NOT NULL, created_run TEXT, usage_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            )""")

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.path)

    def propose(self, label: str, definition: str, *, run_id: str | None = None,
                domain: str | None = None, range_: str | None = None) -> dict:
        normalized = re.sub(r"[^a-z0-9]+", "", label.casefold())
        if not normalized or not definition.strip():
            raise ValueError("dynamic relation requires a label and definition")
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT id, label, definition, status, usage_count FROM relation_registry WHERE normalized_label = ?",
                (normalized,),
            ).fetchone()
            if existing:
                connection.execute("UPDATE relation_registry SET usage_count = usage_count + 1, updated_at = ? WHERE id = ?", (now, existing[0]))
                return {"id": existing[0], "label": existing[1], "definition": existing[2],
                        "status": existing[3], "reused": True, "usage_count": existing[4] + 1}
            relation_id = f"rel-{normalized}"
            connection.execute("INSERT INTO relation_registry VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)",
                               (relation_id, label, normalized, definition, domain, range_, "provisional", run_id, now, now))
        return {"id": relation_id, "label": label, "definition": definition, "status": "provisional", "reused": False, "usage_count": 1}

    def list(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute("SELECT id, label, definition, domain_name, range_name, status, created_run, usage_count, created_at, updated_at FROM relation_registry ORDER BY updated_at DESC").fetchall()
        keys = ("id", "label", "definition", "domain", "range", "status", "created_run", "usage_count", "created_at", "updated_at")
        return [dict(zip(keys, row, strict=True)) for row in rows]

    def review(self, relation_id: str, status: str) -> dict:
        if status not in {"provisional", "approved", "rejected"}:
            raise ValueError("unknown relation review status")
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            cursor = connection.execute("UPDATE relation_registry SET status = ?, updated_at = ? WHERE id = ?", (status, now, relation_id))
            if not cursor.rowcount:
                raise KeyError(relation_id)
        return next(row for row in self.list() if row["id"] == relation_id)

