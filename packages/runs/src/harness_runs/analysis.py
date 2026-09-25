"""Summaries of saved runs for people and agents (used by ``scripts/runs.py``).

Everything here reads only what a run already saved: ``run.json`` (status,
checks, trace, prompt versions) and ``tool_calls.jsonl``. Model-call timings
come from the trace rows the model client writes ("Model call finished ·
<model> · <purpose> · <seconds>s").
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

from harness_schema.models import Run

_SECONDS = re.compile(r"(?:· |after )(\d+(?:\.\d+)?)s\b")
_TIMEOUT = re.compile(r"\(([^·()]+) · ([^()]+)\)")
_ISSUE_ORDER = {"fail": 0, "warn": 1}


@dataclass
class ModelCall:
    model: str
    purpose: str
    seconds: float | None
    ok: bool
    detail: str


@dataclass
class RunSummary:
    run: Run
    model_calls: list[ModelCall] = field(default_factory=list)
    tool_calls: int = 0

    @property
    def main_issue(self) -> str:
        """The most severe check that did not pass, as "name: detail" ("" when all passed)."""
        issues = sorted((c for c in self.run.checks if c.status in _ISSUE_ORDER),
                        key=lambda c: _ISSUE_ORDER[c.status])
        return f"{issues[0].name}: {issues[0].detail}" if issues else ""

    @property
    def model_seconds(self) -> float:
        return round(sum(call.seconds or 0.0 for call in self.model_calls), 1)

    @property
    def failed_model_calls(self) -> int:
        return sum(not call.ok for call in self.model_calls)

    def slowest_calls(self, count: int = 5) -> list[ModelCall]:
        timed = [call for call in self.model_calls if call.seconds is not None]
        return sorted(timed, key=lambda call: call.seconds or 0.0, reverse=True)[:count]

    def seconds_by_purpose(self) -> dict[str, float]:
        totals: Counter[str] = Counter()
        for call in self.model_calls:
            totals[call.purpose] += call.seconds or 0.0
        return {purpose: round(seconds, 1) for purpose, seconds in totals.most_common()}


def parse_model_calls(run: Run) -> list[ModelCall]:
    """Model calls from the trace: one per "Model call finished" or "Model call failed" row."""
    calls = []
    for row in run.trace:
        if row.action not in {"Model call finished", "Model call failed"}:
            continue
        text = row.returned or ""
        timeout = _TIMEOUT.search(text) if text.startswith("timed out") else None
        parts = [timeout.group(1), timeout.group(2)] if timeout else [p.strip() for p in text.split("·")]
        seconds = _SECONDS.search(text)
        calls.append(ModelCall(
            model=parts[0].strip() if parts else "",
            purpose=parts[1].strip() if len(parts) > 1 else "",
            seconds=float(seconds.group(1)) if seconds else None,
            ok=row.action == "Model call finished",
            detail=row.returned or "",
        ))
    return calls


def load_runs(root: Path) -> list[Run]:
    """Every readable run under ``root`` (artifact folders or fixture files), newest first."""
    paths = list(root.glob("*/run.json")) + list(root.glob("r-*.json"))
    runs = []
    for path in paths:
        try:
            runs.append(Run.model_validate_json(path.read_text()))
        except (OSError, ValueError):
            continue
    return sorted(runs, key=lambda run: run.started_at, reverse=True)


def summarize(run: Run, root: Path | None = None) -> RunSummary:
    tool_calls = 0
    if root is not None:
        log = root / run.id / "tool_calls.jsonl"
        if log.exists():
            tool_calls = sum(bool(line.strip()) for line in log.read_text().splitlines())
    return RunSummary(run=run, model_calls=parse_model_calls(run), tool_calls=tool_calls)


def stats_by_path(summaries: list[RunSummary]) -> list[dict[str, object]]:
    """Per path: runs, answered share, median duration and model calls, most common issue."""
    rows = []
    for path in sorted({item.run.path for item in summaries}):
        group = [item for item in summaries if item.run.path == path]
        issues = Counter(item.main_issue.split(":")[0] for item in group if item.main_issue)
        rows.append({
            "path": path,
            "runs": len(group),
            "answered": sum(item.run.status == "answered" for item in group),
            "median_duration_s": round(median(item.run.duration_s for item in group), 1),
            "median_model_calls": median(len(item.model_calls) for item in group),
            "median_model_seconds": round(median(item.model_seconds for item in group), 1),
            "top_issue": issues.most_common(1)[0][0] if issues else "",
        })
    return rows


__all__ = ["ModelCall", "RunSummary", "load_runs", "parse_model_calls", "stats_by_path", "summarize"]
