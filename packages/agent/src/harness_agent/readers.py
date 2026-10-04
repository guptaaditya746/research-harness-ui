"""Parallel paper readers (Phase 3d): short findings per paper, read side by side.

After the supervisor's research, the papers with the most observed evidence (or, when the
supervisor found nothing, the papers of the question's top-ranked passages) are each given to
a reader: one structured call on the agent model (Lite) over that paper's passages that rank
highest for the question and the chosen reading. Readers run in parallel, so reading more
papers costs about one call's time. Each reader returns up to four findings citing only the
passages it was shown; the findings join the supervisor's in the hand-off to the pipeline.

``HARNESS_SUBAGENT_READERS`` (default 3, 0 = off) papers, ``HARNESS_SUBAGENT_PASSAGES``
(default 6) passages each.
"""

from __future__ import annotations

import contextvars
import logging
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from time import monotonic
from typing import Any

from harness_extraction import complete, model_purpose
from harness_prompts import render

from .mcp import validate_findings

PASSAGE_CHARS = 1200
LOGGER = logging.getLogger(__name__)


@dataclass
class ReaderResult:
    paper_id: str
    passage_ids: list[str] = field(default_factory=list)
    findings: list[dict[str, Any]] = field(default_factory=list)
    duration_ms: float = 0.0
    error: str | None = None


def choose_papers(corpus: Any, question: str, weighted_ids: Iterable[tuple[str, int]], limit: int) -> list[str]:
    """Papers ranked by the evidence already found in them, then by the question's top passages."""
    by_passage = getattr(corpus, "by_passage", {}) or {}
    scores: Counter[str] = Counter()
    for passage_id, weight in weighted_ids:
        passage = by_passage.get(passage_id)
        if passage is not None:
            scores[str(passage["paper_id"])] += weight
    chosen = [paper for paper, _ in scores.most_common(limit)]
    if len(chosen) < limit and hasattr(corpus, "hybrid_rank"):
        # Fill the remaining slots with the papers the question itself ranks highest.
        try:
            for row in corpus.hybrid_rank(question, limit=20)["items"]:
                paper = str(row["paper_id"])
                if paper not in chosen:
                    chosen.append(paper)
                if len(chosen) >= limit:
                    break
        except Exception as exc:  # noqa: BLE001 - readers are optional
            LOGGER.warning("Paper choice by question ranking failed: %s", exc)
    return chosen


def _passages(corpus: Any, query: str, paper_id: str, limit: int) -> list[dict[str, Any]]:
    try:
        if hasattr(corpus, "hybrid_rank"):
            return corpus.hybrid_rank(query, scope={"paper_ids": [paper_id]}, limit=limit)["items"]
        return corpus.rank_evidence(query, paper_ids=[paper_id], limit=limit)["items"]
    except Exception:  # noqa: BLE001
        return []


def read_paper(model: Any, corpus: Any, question: str, concept: dict | None, paper_id: str,
               limit: int, rows: list[dict[str, Any]] | None = None) -> ReaderResult:
    """``rows``: the passages to read; by default the paper's passages that rank highest."""
    started = monotonic()
    interpretation = f"{(concept or {}).get('label', '')}: {(concept or {}).get('gloss', '')}".strip(": ")
    if rows is None:
        rows = _passages(corpus, f"{question} {interpretation}".strip(), paper_id, limit)
    result = ReaderResult(paper_id=paper_id, passage_ids=[str(row["passage_id"]) for row in rows])
    if not rows:
        result.error = "no passages"
        return result
    title = getattr(corpus, "paper_title", lambda _pid: None)(paper_id)
    payload = {
        "passages": [{"passage_id": str(row["passage_id"]), "text": str(row.get("text", ""))[:PASSAGE_CHARS]}
                     for row in rows],
        "paper": {"paper_id": paper_id, "title": title},
        "research_interpretation": interpretation or None,
        "question": question,
    }
    try:
        with model_purpose(f"paper reader {paper_id}"):
            reply = complete(model, render("agent/paper_reader", payload))
        result.findings, _ = validate_findings((reply or {}).get("findings"), set(result.passage_ids))
        result.findings = result.findings[:4]
    except Exception as exc:  # noqa: BLE001 - one failed reader must not fail the run
        result.error = f"{type(exc).__name__}: {exc}"[:300]
    result.duration_ms = round((monotonic() - started) * 1000, 1)
    return result


def run_readers(model: Any, corpus: Any, question: str, concept: dict | None, paper_ids: list[str],
                limit: int) -> list[ReaderResult]:
    """Read the papers in parallel; each reader runs in a copy of the caller's context (deadline, events)."""
    if not paper_ids:
        return []
    with ThreadPoolExecutor(max_workers=len(paper_ids), thread_name_prefix="paper-reader") as pool:
        futures = [
            pool.submit(contextvars.copy_context().run, read_paper, model, corpus, question, concept, paper, limit)
            for paper in paper_ids
        ]
        return [future.result() for future in futures]


__all__ = ["ReaderResult", "choose_papers", "read_paper", "run_readers"]
