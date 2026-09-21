"""Deterministic coverage ledger and its monotonic transition engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, TypedDict

ObligationKind = Literal["structural", "content"]
ObligationStatus = Literal["open", "satisfied", "unsupported", "out_of_budget"]


class SearchRecord(TypedDict, total=False):
    tool: str
    query: str
    scope: dict[str, object]


@dataclass
class Obligation:
    id: str
    subject: str
    property: str
    kind: ObligationKind
    expected_object: str | None = None
    status: ObligationStatus = "open"
    attempts: int = 0
    evidence: list[str] = field(default_factory=list)
    searched: list[SearchRecord] = field(default_factory=list)


def _local_name(value: str) -> str:
    return value.rsplit("#", 1)[-1].rsplit("/", 1)[-1]


class Ledger:
    """Own obligations and permit only monotonic transitions out of ``open``."""

    def __init__(self, obligations: list[Obligation]) -> None:
        ids = [item.id for item in obligations]
        if len(ids) != len(set(ids)):
            raise ValueError("obligation ids must be unique")
        self.obligations = obligations

    def _open(self, obligation_id: str) -> Obligation:
        try:
            item = next(item for item in self.obligations if item.id == obligation_id)
        except StopIteration as error:
            raise KeyError(obligation_id) from error
        if item.status != "open":
            raise ValueError(f"obligation {obligation_id} is not open")
        return item

    def next(self, count: int = 1) -> list[Obligation]:
        if count < 0:
            raise ValueError("count must be non-negative")
        return [item for item in self.obligations if item.status == "open"][:count]

    def satisfy(self, obligation_id: str, evidence: list[str]) -> None:
        if not evidence:
            raise ValueError("satisfied obligations require evidence")
        item = self._open(obligation_id)
        item.status = "satisfied"
        item.attempts += 1
        item.evidence = list(dict.fromkeys(evidence))

    def mark_unsupported(self, obligation_id: str, searched: list[SearchRecord]) -> None:
        item = self._open(obligation_id)
        tools = {record.get("tool") for record in searched}
        if item.kind == "content" and not {"search", "grep"}.issubset(tools):
            raise ValueError("unsupported content requires search and grep records")
        if not searched:
            raise ValueError("unsupported obligations require search evidence")
        item.status = "unsupported"
        item.attempts += 1
        item.searched = searched

    def mark_out_of_budget(self, obligation_ids: list[str] | None = None) -> None:
        selected = set(obligation_ids) if obligation_ids is not None else None
        for item in self.obligations:
            if item.status == "open" and (selected is None or item.id in selected):
                item.status = "out_of_budget"
                item.attempts += 1

    def close_structural(self, facts: list[dict[str, object]]) -> None:
        """Close structural obligations against the complete ingest fact table."""
        for item in list(self.obligations):
            if item.status != "open" or item.kind != "structural":
                continue
            subject = _local_name(item.subject)
            predicate = _local_name(item.property)
            matched = [
                str(fact["source_id"])
                for fact in facts
                if str(fact.get("subject")) == subject
                and (
                    str(fact.get("predicate")) == predicate
                    or (predicate == "type" and str(fact.get("predicate")) == "title")
                )
                and (
                    item.expected_object is None
                    or _local_name(str(fact.get("object"))) == _local_name(item.expected_object)
                )
                and fact.get("source_id")
            ]
            if matched:
                self.satisfy(item.id, matched)
            else:
                self.mark_unsupported(
                    item.id,
                    [
                        {
                            "tool": "facts",
                            "query": predicate,
                            "scope": {"paper_ids": [subject]},
                        }
                    ],
                )

    @property
    def closed(self) -> bool:
        return all(item.status != "open" for item in self.obligations)

    @property
    def queryable(self) -> bool:
        return self.closed

    def coverage(self) -> float:
        if not self.obligations:
            return 1.0
        covered = sum(item.status in {"satisfied", "unsupported"} for item in self.obligations)
        return covered / len(self.obligations)

    def as_dict(self) -> dict[str, object]:
        return {
            "obligations": [asdict(item) for item in self.obligations],
            "coverage": self.coverage(),
        }


__all__ = ["Ledger", "Obligation", "ObligationKind", "ObligationStatus", "SearchRecord"]
