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

import json
import re
import threading
from collections import OrderedDict
from difflib import SequenceMatcher
from typing import Any, ClassVar, Literal

from harness_config import settings
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
    "authors_in_venues",
    "papers_with_title_term",
    "papers_in_venue",
    "paper_citation_counts",
    "research",
]

EvidenceFocus = Literal["general", "results", "limitations", "methods"]

# Words that are not part of a venue's name ("ACL 2025 main conference", "long papers").
VENUE_FILLER = {"main", "long", "short", "conference", "papers", "paper", "proceedings", "track"}

# Rankings whose measure an ambiguous word leaves open ("most successful author"): the chosen reading
# decides the measure, so these are planned with the reading.
RANKING_OPERATIONS = {"author_paper_counts", "author_citation_counts", "paper_citation_counts"}
# Operations on one or two named papers: a venue or year in the question describes the paper.
_NAMED_PAPER_OPERATIONS = {"find_paper", "paper_authors", "paper_year", "paper_venue", "cites"}
# Which filters (year, venues, title term) each other fact operation applies.
_FILTERS_USED: dict[str, set[str]] = {
    "papers_in_year": {"year"},
    "papers_with_title_term": {"term", "year"},
    "papers_in_venue": {"venues"},
    "authors_in_venues": {"venues"},
}
_PLACEHOLDERS = {"none", "n/a", "na", "null", "nothing", "-", "[]", "no"}

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
    "authors_in_venues": "authors with a corpus paper in every named venue (DBLP venue facts)",
    "papers_with_title_term": "corpus papers whose title contains the term (DBLP titles)",
    "papers_in_venue": "number of corpus papers published in the venue (DBLP venue facts)",
    "paper_citation_counts": "number of distinct corpus papers citing each paper (resolved references)",
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
    "authors_in_venues": "Authors who have a corpus paper in each of two or more named venues. venues=[venue, venue, ...].",
    "papers_with_title_term": "Corpus papers whose title contains a word or phrase. term=the word or phrase; "
                              "year=YYYY if the question names one.",
    "papers_in_venue": "Count or list the corpus papers published in one named venue. venues=[venue].",
    "paper_citation_counts": "Corpus papers ranked by how many other corpus papers cite them: the most cited "
                             "paper (rank=top) or those cited by at least N corpus papers (min_papers=N).",
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
    venues: list[str] = Field(default_factory=list)
    term: str | None = None
    evidence_focus: EvidenceFocus = "general"
    measure: str | None = None
    ignored: list[str] = Field(default_factory=list)   # question conditions the operation does not apply
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

    @field_validator("venues", mode="before")
    @classmethod
    def _clean_venues(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        return [str(item).strip() for item in value if str(item).strip()][:4]

    @field_validator("ignored", mode="before")
    @classmethod
    def _clean_ignored(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        items = [str(item).strip() for item in value if str(item).strip()]
        return [item for item in items if item.casefold().strip(".") not in _PLACEHOLDERS][:6]

    @field_validator("term", mode="before")
    @classmethod
    def _clean_term(cls, value: Any) -> str | None:
        text = str(value or "").strip().strip("\"'")
        return text or None

    @field_validator("evidence_focus", mode="before")
    @classmethod
    def _focus(cls, value: Any) -> str:
        text = str(value or "general").strip().casefold()
        return text if text in {"general", "results", "limitations", "methods"} else "general"

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
        if self.operation in {"author_paper_counts", "author_citation_counts", "paper_citation_counts"}:
            return self.rank == "top" or (self.min_papers is not None and self.min_papers >= 1)
        if self.operation == "authors_in_venues":
            return len(self.venues) >= 2
        if self.operation == "papers_in_venue":
            return len(self.venues) == 1                 # several venues: research (no silent first-only count)
        if self.operation == "papers_with_title_term":
            return self.term is not None and (self.year is None or 1900 <= self.year <= 2100)
        return True

    def ungrounded(self, question: str) -> list[str]:
        """The arguments of a fact operation that the question does not state.

        A fact operation computes from its arguments alone, so each must come from the question's
        words: a venue, term or paper taken from a reading's label would answer a different question.
        Words match as whole words or word prefixes ("hallucination" / "hallucinations").
        """
        if self.operation == "research":
            return []
        asked = _words(question)
        missing = []
        for paper in self.papers:
            tokens = [word for word in _words(paper) if word not in _STOP]
            if tokens and sum(_stated(word, asked) for word in tokens) / len(tokens) < 0.6:
                missing.append(f"paper {paper!r}")
        for venue in self.venues:
            if not all(_stated(word, asked) for word in _words(venue) if word not in VENUE_FILLER | _STOP):
                missing.append(f"venue {venue!r}")
        if self.term and not all(_stated(word, asked) for word in _words(self.term) if word not in _STOP):
            missing.append(f"term {self.term!r}")
        if self.year is not None and str(self.year) not in asked:
            missing.append(f"year {self.year}")
        return missing

    def unused_filters(self) -> list[str]:
        """Filters the plan carries that its operation does not apply ("papers from 2024 that cite X" as
        citing_count with a year): answering would silently drop that condition. For operations on a
        named paper a venue or year only describes that paper, so they are not filters there."""
        if self.operation == "research" or self.operation in _NAMED_PAPER_OPERATIONS:
            return []
        uses = _FILTERS_USED.get(self.operation, set())
        # A year that is part of a venue ("ACL 2025") is applied with the venue.
        year_in_venue = "venues" in uses and any(str(self.year) in _words(venue) for venue in self.venues)
        present = {"year": self.year is not None and not year_in_venue, "venues": bool(self.venues),
                   "term": self.term is not None}
        return [name for name, given in present.items() if given and name not in uses]

    @classmethod
    def fallback(cls, why: str) -> QueryPlan:
        return cls(operation="research", rationale=why, source="fallback")


def corpus_profile(corpus: Any) -> dict[str, Any]:
    """What the collection holds, for the planner: paper counts per venue and per year (DBLP facts).
    A condition every corpus paper meets is not a condition to apply; the planner needs this to tell."""
    papers = getattr(corpus, "by_paper", {}) or {}
    venues: dict[str, set[str]] = {}
    years: dict[str, set[str]] = {}
    for row in getattr(corpus, "fact_rows", []) or []:
        subject = str(row.get("subject"))
        if subject not in papers:
            continue
        if row.get("predicate") == "venue":
            venues.setdefault(str(row.get("object")), set()).add(subject)
        elif row.get("predicate") == "year":
            years.setdefault(str(row.get("object")), set()).add(subject)
    top = sorted(venues.items(), key=lambda item: -len(item[1]))[:12]
    return {"papers": len(papers), "venues": {name: len(ids) for name, ids in top},
            "years": {year: len(ids) for year, ids in sorted(years.items())}}


class QueryPlanner:
    """Model-backed question understanding with strict validation."""

    # Plans of the bare question (no reading), shared by all readings of one question in this process:
    # a fact lookup's arguments come from the question alone, so every reading gets the same plan.
    _question_plans: ClassVar[OrderedDict[tuple[str, ...], QueryPlan]] = OrderedDict()
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, model: Any, profile: dict[str, Any] | None = None) -> None:
        self.model = model
        self.profile = profile

    def plan_question(self, question: str) -> QueryPlan:
        """``plan(question)`` without a reading, cached per model, prompt set, corpus and question."""
        key = (str(getattr(self.model, "model", type(self.model).__name__)), settings().prompt_set or "v2",
               json.dumps(self.profile, sort_keys=True), " ".join(question.split()))
        with self._lock:
            cached = self._question_plans.get(key)
        if cached is not None:
            return cached
        plan = self.plan(question)
        if plan.source == "model":                                   # a failed call is not remembered
            with self._lock:
                self._question_plans[key] = plan
                while len(self._question_plans) > 256:
                    self._question_plans.popitem(last=False)
        return plan

    def plan(self, question: str, concept: dict[str, Any] | None = None) -> QueryPlan:
        request: dict[str, Any] = {"question": question}
        if self.profile and (settings().prompt_set or "v2") != "v1":
            request["corpus"] = self.profile
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
        missing = plan.ungrounded(question)
        unused = plan.unused_filters()
        if missing:
            plan = QueryPlan.fallback(f"plan for {plan.operation} uses {', '.join(missing)}, "
                                      "which the question does not state")
        elif unused:
            plan = QueryPlan.fallback(f"{plan.operation} cannot apply the {' and '.join(unused)} the question names")
        elif plan.operation != "research" and plan.ignored:
            plan = QueryPlan.fallback(f"{plan.operation} would leave out part of the question "
                                      f"({'; '.join(plan.ignored[:3])})")
        if concept and concept.get("label") and plan.operation == "research":
            # The user chose this reading; the answer must say it used it, not
            # whatever measure the model derived from the question's wording.
            plan = plan.model_copy(update={"measure": str(concept["label"])})
        return plan


# ─── Paper resolution ────────────────────────────────────────────────────────

_STOP = {"the", "a", "an", "paper", "papers", "work", "article", "study", "of", "and", "for",
         "on", "in", "with", "to", "by", "its"}


def _words(text: object) -> list[str]:
    """Lower-case words and digits; "ACL2025" reads as "ACL 2025"."""
    return re.findall(r"[a-z0-9]+", re.sub(r"(?<=[A-Za-z])(?=(?:19|20)\d{2}\b)", " ", str(text)).casefold())


def _stated(word: str, asked: list[str]) -> bool:
    """Whether ``word`` occurs in the question as a word or a word prefix, either way round
    ("model" / "models", "hallucinations" / "hallucination")."""
    return any(word == other or (len(word) >= 3 and other.startswith(word))
               or (len(other) >= 4 and word.startswith(other) and len(word) - len(other) <= 2) for other in asked)


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


__all__ = ["OPERATIONS", "RANKING_OPERATIONS", "VENUE_FILLER", "PaperResolver", "QueryPlan", "QueryPlanner", "corpus_profile"]
