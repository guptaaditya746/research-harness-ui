"""Small, versioned research playbooks disclosed to the agent on demand."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchSkill:
    name: str
    description: str
    instructions: str
    version: str = "1"


SKILLS = (
    ResearchSkill(
        "numeric-result-verification",
        "Verify a reported metric, value, comparator, benchmark, and setting.",
        "Search result passages and tables. Preserve metric, value, unit, comparator, benchmark, and setting separately. Treat absent context as unresolved, not inferred.",
    ),
    ResearchSkill(
        "corpus-aggregation",
        "Answer counts, rankings and earliest/latest questions over all papers.",
        "Call aggregate_facts once (count_papers_by_author, count_by_value, min, max, "
        "count_incoming_citations, count_citations_by_author) or metadata_table for the raw facts. "
        "Do not open papers one by one. Cite the fact or reference IDs the aggregate returns.",
    ),
    ResearchSkill(
        "citation-analysis",
        "Analyse corpus-local incoming or outgoing citations.",
        "Use citations with indexed_only=true for corpus-scoped questions. Only report corpus_absent after an exhaustive incoming-index query with no failures.",
    ),
    ResearchSkill(
        "cross-paper-comparison",
        "Compare source-specific claims without merging their provenance.",
        "Create separate claims per source. Compare material context before calling values conflicting; make corpus-level conclusions explicit synthesis claims.",
    ),
    ResearchSkill(
        "corpus-absence-verification",
        "Distinguish exhaustive structured absence from semantic no-evidence-found.",
        "Record evidence surface, scope, coverage, failures, and whether retrieval was exhaustive. Never turn partial semantic search into corpus_absent.",
    ),
    ResearchSkill(
        "limitation-analysis",
        "Find source-backed limitations and their conditions.",
        "Search limitations and neighbouring sections. Keep the affected method, condition, and task attached to each LimitationClaim.",
    ),
    ResearchSkill(
        "claim-conflict-analysis",
        "Resolve possible conflicts by comparing claim context.",
        "Check subject, metric, benchmark, protocol, unit, comparator, and setting. Missing material context leaves a possibleConflict unresolved.",
    ),
    ResearchSkill(
        "table-result-verification",
        "Ground claims in table cells with their headers and captions.",
        "Keep table ID, row, column, caption, page, and cell value. A number without its contextual headers is insufficient evidence.",
    ),
)


def catalog() -> str:
    return "; ".join(f"{skill.name}: {skill.description}" for skill in SKILLS)


def relevant(question: str) -> list[ResearchSkill]:
    text = question.casefold()
    keywords = {
        "numeric-result-verification": ("%", "improvement", "accuracy", "f1", "score", "result"),
        "corpus-aggregation": ("how many", "most", "least", "earliest", "latest", "oldest", "newest",
                               "best", "rank", "more than", "count", "every paper", "all papers"),
        "citation-analysis": ("cite", "citation", "citing", "cited"),
        "cross-paper-comparison": ("compare", "across", "shared", "common"),
        "corpus-absence-verification": ("any", "none", "no paper", "absent"),
        "limitation-analysis": ("limitation", "weakness", "fails", "failure"),
        "claim-conflict-analysis": ("conflict", "contradict", "disagree", "different result"),
        "table-result-verification": ("table", "figure", "metric", "benchmark"),
    }
    selected = [skill for skill in SKILLS if any(term in text for term in keywords[skill.name])]
    return selected[:3]
