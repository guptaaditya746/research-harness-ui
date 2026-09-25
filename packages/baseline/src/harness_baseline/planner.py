"""Question understanding: turn any phrasing into a typed query plan.

The research pipeline used to route questions by matching literal phrases
("who authored", "how many indexed papers cite", ...). Any other wording of
the same question silently took the slow full-text path. Here a model reads
the question once and fills in a small typed plan; paper mentions are then
resolved against corpus titles, and deterministic operations answer from
verified facts. Phrasing no longer decides the route; meaning does.

Design rules:
- The model only *classifies and extracts*; it never supplies facts.
- Every plan is validated. An invalid or unavailable plan degrades to
  ``research`` (the full-text pipeline), never to a guessed answer.
- A fact operation whose paper cannot be resolved also degrades to
  ``research`` and says so in the trace.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Literal

from harness_extraction import complete, model_purpose
from harness_prompts import Prompt, render
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

Operation = Literal[
    "find_paper",
    "paper_authors",
    "paper_year",
    "paper_venue",
    "citing_papers",
    "citing_count",
    "cited_papers",
    "cites",
    "papers_in_year",
    "publication_year_extreme",
    "author_paper_counts",
    "author_citation_counts",
    "research",
]

EvidenceFocus = Literal["general", "results", "limitations", "methods", "metrics"]

# What each fact operation measures, stated with the answer so an ambiguous
# question ("best author") never silently answers a different measure.
MEASURES: dict[str, str] = {
    "find_paper": "the corpus paper matching the named method or title",
    "paper_authors": "the paper's authors (DBLP)",
    "paper_year": "the paper's publication year (DBLP)",
    "paper_venue": "the paper's publication venue (DBLP)",
    "citing_papers": "corpus papers whose resolved references cite the paper",
    "citing_count": "number of distinct corpus papers citing the paper",
    "cited_papers": "corpus papers the paper's resolved references point to",
    "cites": "whether a resolved reference links the two papers",
    "papers_in_year": "number of corpus papers with that publication year",
    "publication_year_extreme": "earliest or latest publication year among corpus papers",
    "author_paper_counts": "number of distinct corpus papers per author (a preprint and its journal version count once)",
    "author_citation_counts": "number of distinct corpus papers citing each author's papers",
}

OPERATIONS: dict[str, str] = {
    "find_paper": "Identify which corpus paper a name or description refers to "
                  "(e.g. which paper introduced / proposed / presents X). papers=[X].",
    "paper_authors": "List the authors of one paper. papers=[paper].",
    "paper_year": "Publication year of one paper. papers=[paper].",
    "paper_venue": "Publication venue of one paper. papers=[paper].",
    "citing_papers": "Which corpus papers cite a given paper. papers=[cited paper].",
    "citing_count": "How many corpus papers cite a given paper. papers=[cited paper].",
    "cited_papers": "Which corpus papers a given paper cites. papers=[citing paper].",
    "cites": "Whether one paper cites another. papers=[citing paper, cited paper].",
    "papers_in_year": "Count or list corpus papers published in a year. year=YYYY.",
    "publication_year_extreme": "Earliest or latest publication year in the corpus and its "
                                "papers. extreme=earliest|latest.",
    "author_paper_counts": "Authors ranked by number of distinct corpus papers: the authors with "
                           "the most papers (rank=top) or with at least N papers (min_papers=N).",
    "author_citation_counts": "Authors ranked by how often other corpus papers cite their papers: the most "
                              "cited authors (rank=top) or those with at least N citing papers (min_papers=N).",
    "research": "Anything that needs the papers' full text: results, numbers, methods, "
                "mechanisms, limitations, comparisons, opinions, open questions.",
}

def planning_prompt(request: dict[str, Any], prompt_set: str | None = None) -> Prompt:
    """The question-understanding prompt (``planning/query_plan``) for one request."""
    return render("planning/query_plan", request, prompt_set=prompt_set, operations=OPERATIONS)


class QueryPlan(BaseModel):
    model_config = ConfigDict(extra="ignore")

    operation: Operation = "research"
    papers: list[str] = Field(default_factory=list)
    year: int | None = None
    extreme: Literal["earliest", "latest"] | None = None
    rank: Literal["top"] | None = None
    min_papers: int | None = None
    evidence_focus: EvidenceFocus = "general"
    benchmark: str | None = None
    measure: str | None = None
    rationale: str = ""
    source: Literal["model", "fallback"] = "model"

    def stated_measure(self) -> str | None:
        """The measure the answer uses: fixed for fact operations, the planner's phrase for research."""
        return MEASURES.get(self.operation) or (self.measure.strip() if self.measure else None)

    @field_validator("papers", mode="before")
    @classmethod
    def _clean_papers(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        return [str(item).strip() for item in value if str(item).strip()][:2]

    @field_validator("evidence_focus", mode="before")
    @classmethod
    def _focus(cls, value: Any) -> str:
        text = str(value or "general").strip().casefold()
        return text if text in {"general", "results", "limitations", "methods", "metrics"} else "general"

    def requirements_met(self) -> bool:
        """Whether the plan carries the arguments its operation needs."""
        needs_papers = {
            "find_paper": 1, "paper_authors": 1, "paper_year": 1, "paper_venue": 1,
            "citing_papers": 1, "citing_count": 1, "cited_papers": 1, "cites": 2,
        }
        if len(self.papers) < needs_papers.get(self.operation, 0):
            return False
        if self.operation == "papers_in_year":
            return self.year is not None and 1900 <= self.year <= 2100
        if self.operation == "publication_year_extreme":
            return self.extreme is not None
        if self.operation in {"author_paper_counts", "author_citation_counts"}:
            return self.rank == "top" or (self.min_papers is not None and self.min_papers >= 1)
        return True

    @classmethod
    def fallback(cls, why: str) -> QueryPlan:
        return cls(operation="research", rationale=why, source="fallback")


class QueryPlanner:
    """Model-backed question understanding with strict validation."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def plan(self, question: str, concept: dict[str, Any] | None = None) -> QueryPlan:
        request: dict[str, Any] = {"question": question}
        if concept:
            request["research_interpretation"] = {
                "label": concept.get("label"), "description": concept.get("gloss"),
            }
        try:
            with model_purpose("question understanding"):
                payload = complete(self.model, planning_prompt(request))
        except Exception as exc:  # noqa: BLE001 - planning must never fail a run
            return QueryPlan.fallback(f"planner unavailable: {type(exc).__name__}")
        try:
            plan = QueryPlan.model_validate(payload)
        except ValidationError as exc:
            return QueryPlan.fallback(f"invalid plan: {exc.errors()[0].get('msg')}")
        if not plan.requirements_met():
            return QueryPlan.fallback(f"plan for {plan.operation} lacks required arguments")
        return plan


# ─── Paper resolution ────────────────────────────────────────────────────────

_STOP = {"the", "a", "an", "paper", "papers", "work", "article", "study", "of", "and", "for",
         "on", "in", "with", "to", "by", "its"}


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text).casefold()))


def _tokens(text: str) -> set[str]:
    return {word for word in _norm(text).split() if word not in _STOP and len(word) > 1}


class PaperResolver:
    """Resolve a paper mention to one corpus paper id, or refuse when ambiguous."""

    def __init__(self, titles: dict[str, str]) -> None:
        self.titles = {pid: title for pid, title in titles.items() if title}

    def resolve(self, mention: str) -> str | None:
        mention_norm = _norm(mention)
        mention_norm = " ".join(w for w in mention_norm.split() if w not in {"the", "paper"})
        if not mention_norm:
            return None
        exact = [pid for pid, title in self.titles.items() if _norm(title) == mention_norm]
        if len(exact) == 1:
            return exact[0]
        # A mention that is a leading part of the title ("FOCUS", "Dream to
        # Manipulate", the part before a colon).
        heads = [
            pid for pid, title in self.titles.items()
            if _norm(title.split(":", 1)[0]) == mention_norm
            or (len(mention_norm.split()) >= 2 and _norm(title).startswith(mention_norm + " "))
        ]
        if len(heads) == 1:
            return heads[0]
        wanted = _tokens(mention)
        if not wanted:
            return None
        scored = []
        for pid, title in self.titles.items():
            have = _tokens(title)
            coverage = len(wanted & have) / len(wanted)
            similarity = SequenceMatcher(None, mention_norm, _norm(title)).ratio()
            scored.append((max(coverage, similarity), pid))
        scored.sort(reverse=True)
        best_score, best = scored[0]
        runner_up = scored[1][0] if len(scored) > 1 else 0.0
        if best_score >= 0.75 and best_score - runner_up >= 0.15:
            return best
        return None


__all__ = ["OPERATIONS", "PaperResolver", "QueryPlan", "QueryPlanner"]
