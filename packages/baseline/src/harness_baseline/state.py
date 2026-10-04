"""Pipeline state, results and the outcomes that end a research run early."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, TypedDict

from harness_extraction import (
    AnswerResult,
    ExtractedTriple,
    SchemaInference,
)
from harness_schema.models import (
    Run,
    TraceRow,
)

from .planner import QueryPlan

TraceSink = Callable[[TraceRow], None]


class BaselineState(TypedDict, total=False):
    question: str
    concept: dict[str, Any] | None
    agent_findings: list[dict[str, Any]]
    plan: QueryPlan
    paper_ids: list[str]
    computed: list[dict[str, Any]]
    passages: list[dict[str, Any]]
    inferred: SchemaInference
    extracted: list[ExtractedTriple]
    answer: AnswerResult
    retrieval: dict[str, Any]
    extraction_batches: list[dict[str, Any]]
    evidence_graph: list[dict[str, Any]]


class BaselineResult(TypedDict):
    run: Run
    artifacts: dict[str, bytes]


class NoResults(RuntimeError):
    pass


class CorpusAbsent(NoResults):
    def __init__(self, message: str, evidence_surface: str, inspected: int):
        super().__init__(message)
        self.evidence_surface = evidence_surface
        self.inspected = inspected


class NoEvidenceFound(NoResults):
    """A completed semantic search, deliberately weaker than corpus absence."""
    def __init__(self, message: str, coverage: dict[str, Any]):
        super().__init__(message)
        self.coverage = coverage


def retrieval_dsl(question: str) -> str:
    stop = {
        "about",
        "after",
        "before",
        "could",
        "does",
        "from",
        "have",
        "paper",
        "papers",
        "that",
        "their",
        "these",
        "this",
        "what",
        "when",
        "where",
        "which",
        "with",
        "would",
    }
    terms = []
    for term in re.findall(r"[A-Za-z][A-Za-z0-9_-]+", question):
        if (
            len(term) >= 4
            and term.casefold() not in stop
            and term.casefold() not in {t.casefold() for t in terms}
        ):
            terms.append(term)
    if not terms:
        raise NoResults("question has no searchable terms")
    return " OR ".join(terms[:12])

