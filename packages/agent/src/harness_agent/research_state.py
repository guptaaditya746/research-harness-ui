"""Domain state retained when an agent conversation is compacted."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.messages.utils import count_tokens_approximately


def _handles(value: Any) -> set[str]:
    if isinstance(value, dict):
        found = {
            str(item)
            for key, item in value.items()
            if key.endswith("_id") and isinstance(item, (str, int))
        }
        for item in value.values():
            found.update(_handles(item))
        return found
    if isinstance(value, list):
        return set().union(*(_handles(item) for item in value)) if value else set()
    return set()


@dataclass
class ResearchLedger:
    question: str
    interpretation: str | None = None
    findings: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    evidence_ids: set[str] = field(default_factory=set)
    tool_calls: int = 0
    checkpoints: int = 0

    def record(self, tool: str, result: Any, purpose: str) -> None:
        self.tool_calls += 1
        self.evidence_ids.update(_handles(result))
        count = len(result.get("items", [])) if isinstance(result, dict) else None
        self.findings.append(f"{tool}: {count if count is not None else 'completed'} candidates ({purpose})")

    def checkpoint(self, messages: list[dict[str, Any]]) -> dict[str, Any] | None:
        approximate = count_tokens_approximately(
            [HumanMessage(content=json.dumps(message, ensure_ascii=False, default=str)) for message in messages]
        )
        if approximate < 40_000:
            return None
        self.checkpoints += 1
        return {
            "checkpoint": self.checkpoints,
            "before_tokens": approximate,
            "preserved": {
                "question": self.question,
                "interpretation": self.interpretation,
                "findings": self.findings[-20:],
                "unresolved": self.unresolved,
                "evidence_ids": sorted(self.evidence_ids),
            },
            "archived_tool_outputs": max(0, self.tool_calls - 4),
        }

    def summary(self) -> str:
        return json.dumps({
            "question": self.question,
            "findings": self.findings[-20:],
            "unresolved": self.unresolved,
            "evidence_ids": sorted(self.evidence_ids),
        }, ensure_ascii=False)
