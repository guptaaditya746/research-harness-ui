"""Versioned local store for accepted, source-attributed claim subgraphs."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from harness_schema.models import ClaimSubgraph


class ClaimStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS claims (id TEXT PRIMARY KEY, body TEXT NOT NULL)")
            connection.execute("CREATE VIRTUAL TABLE IF NOT EXISTS claims_fts USING fts5(id UNINDEXED, content)")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def upsert(self, claims: list[ClaimSubgraph]) -> None:
        with self._connect() as connection:
            for claim in claims:
                if claim.status != "accepted":
                    continue
                previous = connection.execute(
                    "SELECT body FROM claims WHERE id = ?", (claim.id,)
                ).fetchone()
                if previous:
                    existing = ClaimSubgraph.model_validate_json(previous[0])
                    sources = sorted(set(existing.source_ids) | set(claim.source_ids))
                    claim = claim.model_copy(update={
                        "source_ids": sources,
                        "evidence": {**existing.evidence, **claim.evidence},
                        "fields": {
                            key: field.model_copy(update={
                                "source_ids": sorted(set(field.source_ids) |
                                                     set(existing.fields.get(key, field).source_ids)),
                            }) for key, field in claim.fields.items()
                        },
                    })
                connection.execute(
                    "INSERT OR REPLACE INTO claims(id, body) VALUES (?, ?)",
                    (claim.id, claim.model_dump_json(by_alias=True)),
                )
                connection.execute("DELETE FROM claims_fts WHERE id = ?", (claim.id,))
                content = " ".join([claim.kind, claim.paper_id, *(
                    field.value for field in claim.fields.values()
                )])
                connection.execute(
                    "INSERT INTO claims_fts(id, content) VALUES (?, ?)", (claim.id, content)
                )

    def search(self, question: str, limit: int = 20) -> list[ClaimSubgraph]:
        terms = list(dict.fromkeys(
            term.casefold() for term in re.findall(r"[A-Za-z][A-Za-z0-9]{3,}", question)
        ))
        if not terms:
            return []
        expression = " OR ".join(f'"{term}"' for term in terms)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT claims.body FROM claims_fts JOIN claims ON claims.id = claims_fts.id "
                "WHERE claims_fts MATCH ? ORDER BY bm25(claims_fts) LIMIT ?",
                (expression, limit),
            ).fetchall()
        return [ClaimSubgraph.model_validate_json(row[0]) for row in rows]


__all__ = ["ClaimStore"]
