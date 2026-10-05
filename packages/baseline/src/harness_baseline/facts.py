"""Fact operations: answer metadata and citation questions from verified DBLP facts.

A mixin of ``Baseline20``: methods use the pipeline's state (``self``).
"""

from __future__ import annotations

import re
from functools import cached_property
from typing import Any

from harness_extraction import (
    AnswerResult,
    ExtractedTriple,
)

from .planner import VENUE_FILLER, PaperResolver, QueryPlan
from .state import (
    CorpusAbsent,
)


def normal_text(text: object) -> str:
    """Lower-case words and digits joined by single spaces (the same rule as labelling's ``norm``)."""
    return " ".join(re.findall(r"[a-z0-9]+", str(text).casefold()))


def venue_matches(venue_fact: object, paper_year: object, mention: str) -> bool:
    """Whether a venue fact matches a venue as the question names it.

    The venue name must appear as whole words in the fact ("ACL" matches "ACL 2025", "ACL (1)" and
    "Findings of ACL", not "NAACL"); a year in the mention must appear in the fact or be the paper's
    publication year (DBLP often keeps the year out of the venue string). labelling computes gold
    with the same rule (``structural.venue_matches``).
    """
    # "ACL2025" reads as "ACL 2025"; words like "main conference" or "long papers" are not part of the name.
    words = normal_text(re.sub(r"(?<=[A-Za-z])(?=(?:19|20)\d{2}\b)", " ", str(mention))).split()
    years = [word for word in words if re.fullmatch(r"(?:19|20)\d{2}", word)]
    name = " ".join(word for word in words if word not in years and word not in VENUE_FILLER)
    fact = f" {normal_text(venue_fact)} "
    if name and f" {name} " not in fact:
        return False
    return all(f" {year} " in fact or year == str(paper_year or "") for year in years)


def title_has_term(title: object, term: str) -> bool:
    """Word-prefix match: "hallucination" finds "Hallucinations", "manipulation" not "Manipulate"."""
    wanted = normal_text(term)
    return bool(wanted) and f" {wanted}" in f" {normal_text(title)}"


FactResult = tuple[AnswerResult, list[ExtractedTriple]] | None


class _Scope:
    """What one fact operation reads: the plan, its papers, and the corpus facts (derived lists computed once)."""

    def __init__(self, corpus: Any, plan: QueryPlan, papers: list[str], target: dict[str, Any] | None) -> None:
        self.corpus, self.plan, self.papers, self.target = corpus, plan, papers, target
        self.facts = corpus.seed_facts()

    @cached_property
    def years(self) -> list[dict[str, Any]]:
        return [row for row in self.facts if row["predicate"] == "year" and str(row["object"]).isdigit()]

    @cached_property
    def year_of(self) -> dict[str, str]:
        return {str(row["subject"]): str(row["object"]) for row in self.years}

    @cached_property
    def venue_rows(self) -> list[dict[str, Any]]:
        # Only indexed papers (papers.jsonl) count; labelling computes gold over the same set.
        return [row for row in self.facts if row["predicate"] == "venue" and str(row["subject"]) in self.corpus.by_paper]

    def venue_match(self, row: dict[str, Any], venue: str) -> bool:
        return venue_matches(row["object"], self.year_of.get(str(row["subject"])), venue)


class FactAnswers:
    # ─── Fact operations (typed; selected by the query plan, not by wording) ───

    def _titles(self) -> dict[str, str]:
        titles = {
            str(row["subject"]): str(row["object"])
            for row in self.corpus.seed_facts() if row.get("predicate") == "title"
        }
        for paper_id, paper in self.corpus.by_paper.items():
            titles.setdefault(str(paper_id), str(paper.get("title") or ""))
        return titles

    def _title_row(self, paper_id: str) -> dict[str, Any]:
        for row in self.corpus.seed_facts():
            if row.get("predicate") == "title" and str(row["subject"]) == paper_id:
                return row
        title = self.corpus.paper_title(paper_id) or paper_id
        return {"subject": paper_id, "object": title, "source_id": f"{paper_id}-metadata"}

    def _resolve_papers(self, plan: QueryPlan) -> tuple[list[str], list[str]]:
        """Resolve plan paper mentions; returns (paper ids, unresolved mentions)."""
        resolver = PaperResolver(self._titles())
        resolved, missing = [], []
        for mention in plan.papers:
            paper_id = resolver.resolve(mention)
            (resolved if paper_id else missing).append(paper_id or mention)
        return resolved, missing

    def _fact_query(
        self, plan: QueryPlan, papers: list[str]
    ) -> tuple[AnswerResult, list[ExtractedTriple]] | None:
        """Execute a metadata/citation/aggregation plan over verified facts.

        Returns None when the facts needed for the operation are not indexed (or the plan lacks
        what the operation needs), so the caller can fall back to full-text research. Raises
        CorpusAbsent when the operation completed and established absence on its surface.
        """
        # Conclusions computed below (counts, extremes) with the facts they rest on.
        self._computed: list[dict[str, Any]] = []
        handler = FACT_OPERATIONS.get(plan.operation)
        if handler is None:
            return None
        return handler(self, _Scope(self.corpus, plan, papers, self._title_row(papers[0]) if papers else None))

    # One method per operation (FACT_OPERATIONS); each returns None when the plan lacks what it needs.

    def _compute(self, label: str, operation: str, role: str, value: str, support: list[dict[str, Any]]) -> None:
        self._computed.append({"label": label, "operation": operation,
                               "about": {"role": role, "value": value}, "support": support})

    def _find_paper(self, scope: _Scope) -> FactResult:
        target = scope.target
        if not target:
            return None
        triple = ExtractedTriple(
            subject=str(target["subject"]), predicate="title", object=str(target["object"]),
            source_id=str(target["source_id"]),
            evidence=f"Corpus fact {target.get('fact_id', target['source_id'])}.",
        )
        canonical = self.corpus.canonical_paper_iri(str(target["subject"]))
        return (
            AnswerResult(
                sparql="PREFIX dblp: <https://dblp.org/rdf/schema#>\nSELECT ?paper ?title WHERE { ?paper dblp:title ?title . }",
                answer=f"{target['object']} ({canonical or target['subject']}) [{target['source_id']}].",
                cited_source_ids=[str(target["source_id"])],
            ),
            [triple],
        )

    def _paper_authors(self, scope: _Scope) -> FactResult:
        target = scope.target
        triples = self._author_triples(scope.papers[0]) if target else []
        if not triples:
            return None
        names = [item.object for item in triples]
        return (
            AnswerResult(
                sparql="PREFIX dblp: <https://dblp.org/rdf/schema#>\nSELECT ?author WHERE { ?paper dblp:authoredBy ?author . }",
                answer=f"{target['object']} was authored by {', '.join(names)} [{target['source_id']}].",
                cited_source_ids=sorted({item.source_id for item in triples}),
            ),
            triples,
        )

    def _paper_year_or_venue(self, scope: _Scope) -> FactResult:
        if not scope.target:
            return None
        predicate = "year" if scope.plan.operation == "paper_year" else "venue"
        rows = [row for row in scope.facts if row["subject"] == scope.papers[0] and row["predicate"] == predicate]
        if not rows:
            return None
        return self._fact_answer([self._fact_triple(row) for row in rows],
                                 f"{scope.target['object']} was published in {rows[0]['object']} [{rows[0]['source_id']}].")

    def _citing_papers(self, scope: _Scope) -> FactResult:
        target, paper = scope.target, scope.papers[0] if scope.papers else None
        if not target:
            return None
        rows = [row for row in self.corpus.references if row.get("resolved_paper_id") == paper]
        if not rows:
            raise CorpusAbsent(
                f"No in-corpus evidence found that a paper cites {target['object']}. "
                "This does not establish that no external citation exists.",
                "citation_index", len(self.corpus.references),
            )
        triples = [self._citation_triple(row) for row in rows]
        sources = ", ".join(item.source_id for item in triples)
        if scope.plan.operation == "citing_count":
            count = len({item.subject for item in triples})
            self._compute(f"{count} citing papers", "count distinct in-corpus papers citing it",
                          "subject", self._iri(paper), [{"p": "cites", "o": self._iri(paper)}])
            return self._fact_answer(triples, f"{count} indexed papers cite {target['object']} [{sources}].")
        names = sorted({self.corpus.paper_title(item.subject) or item.subject for item in triples})
        return self._fact_answer(triples, f"{', '.join(names)} cite {target['object']} [{sources}].")

    def _cited_papers(self, scope: _Scope) -> FactResult:
        target = scope.target
        if not target:
            return None
        rows = [row for row in self.corpus.references
                if row["paper_id"] == scope.papers[0] and row.get("resolved_paper_id")]
        if not rows:
            raise CorpusAbsent(f"No resolved in-corpus citations found for {target['object']}.",
                               "citation_index", len(self.corpus.references))
        triples = [self._citation_triple(row) for row in rows]
        names = [self.corpus.paper_title(item.object) or item.object for item in triples]
        return self._fact_answer(triples, f"{target['object']} cites these indexed papers: {', '.join(names)} "
                                          f"[{', '.join(item.source_id for item in triples)}].")

    def _cites(self, scope: _Scope) -> FactResult:
        target, papers = scope.target, scope.papers
        if len(papers) < 2 or not target:
            return None
        cited = self._title_row(papers[1])
        rows = [row for row in self.corpus.references
                if row["paper_id"] == papers[0] and row.get("resolved_paper_id") == papers[1]]
        if not rows:
            raise CorpusAbsent(f"No resolved in-corpus citation from {target['object']} to {cited['object']}.",
                               "citation_index", len(self.corpus.references))
        triples = [self._citation_triple(row) for row in rows]
        return self._fact_answer(triples, f"Yes. {target['object']} cites {cited['object']} "
                                          f"[{', '.join(item.source_id for item in triples)}].")

    def _papers_in_year(self, scope: _Scope) -> FactResult:
        year = scope.plan.year
        if year is None:
            return None
        rows = [row for row in scope.years if int(row["object"]) == year]
        if not rows:
            raise CorpusAbsent(f"No indexed paper has publication year {year}.", "metadata_facts", len(scope.years))
        triples = [self._fact_triple(row) for row in rows]
        names = [self.corpus.paper_title(item.subject) or item.subject for item in triples]
        count = len({item.subject for item in triples})
        self._compute(f"{count} papers in {year}", "count papers whose publication year matches",
                      "value", str(year), [{"p": "year", "o": str(year)}])
        return self._fact_answer(triples, f"{count} corpus papers were published in {year}: {', '.join(names)} "
                                          f"[{', '.join(item.source_id for item in triples)}].")

    def _papers_with_title_term(self, scope: _Scope) -> FactResult:
        plan = scope.plan
        if not plan.term:
            return None
        titles = [row for row in scope.facts
                  if row["predicate"] == "title" and str(row["subject"]) in self.corpus.by_paper]
        rows = [row for row in titles if title_has_term(row["object"], plan.term)
                and (plan.year is None or scope.year_of.get(str(row["subject"])) == str(plan.year))]
        when = f" from {plan.year}" if plan.year is not None else ""
        if not rows:
            raise CorpusAbsent(f"No indexed paper{when} has \"{plan.term}\" in its title.",
                               "metadata_facts", len(titles))
        triples = [self._fact_triple(row) for row in rows]
        self._compute(f"{len(rows)} titles with \"{plan.term}\"",
                      "filter corpus papers whose title contains the term" + (f" and year is {plan.year}" if when else ""),
                      "value", plan.term, [{"p": "title"}])
        return self._fact_answer(triples, f"{len(rows)} indexed papers{when} have \"{plan.term}\" in their title: "
                                 + "; ".join(str(row["object"]) for row in rows)
                                 + f" [{', '.join(item.source_id for item in triples)}].")

    def _papers_in_venue(self, scope: _Scope) -> FactResult:
        if not scope.plan.venues:
            return None
        venue = scope.plan.venues[0]
        rows = [row for row in scope.venue_rows if scope.venue_match(row, venue)]
        if not rows:
            raise CorpusAbsent(f"No indexed paper has a venue matching {venue}.", "metadata_facts",
                               len(scope.venue_rows))
        triples = [self._fact_triple(row) for row in rows]
        found = sorted({str(row["subject"]) for row in rows})
        self._compute(f"{len(found)} papers in {venue}",
                      "count corpus papers whose venue matches (whole words; year from the venue or the paper)",
                      "value", venue, [{"p": "venue"}])
        names = [self.corpus.paper_title(paper) or paper for paper in found]
        return self._fact_answer(triples, f"{len(found)} indexed papers were published in {venue}: "
                                 + "; ".join(names) + f" [{', '.join(item.source_id for item in triples)}].")

    def _authors_in_venues(self, scope: _Scope) -> FactResult:
        venues = scope.plan.venues
        if len(venues) < 2:
            return None
        authors = [row for row in scope.facts
                   if row["predicate"] == "author" and str(row["subject"]) in self.corpus.by_paper]
        papers_by_venue = {venue: {str(row["subject"]) for row in scope.venue_rows if scope.venue_match(row, venue)}
                           for venue in venues}
        shared = set.intersection(*({str(row["object"]) for row in authors if str(row["subject"]) in members}
                                    for members in papers_by_venue.values()))
        if not shared:
            raise CorpusAbsent(f"No author has an indexed paper in all of {', '.join(venues)}.",
                               "metadata_facts", len(scope.venue_rows))
        in_scope = set().union(*papers_by_venue.values())
        rows = [row for row in authors if str(row["object"]) in shared and str(row["subject"]) in in_scope]
        rows += [row for row in scope.venue_rows if str(row["subject"]) in {str(item["subject"]) for item in rows}]
        triples = [self._fact_triple(row) for row in rows]
        lines = []
        for name in sorted(shared):
            parts = []
            for venue, members in papers_by_venue.items():
                own = sorted(str(row["subject"]) for row in authors
                             if str(row["object"]) == name and str(row["subject"]) in members)
                parts.append(f"{venue}: " + ", ".join(self.corpus.paper_title(paper) or paper for paper in own))
            lines.append(f"{name} ({'; '.join(parts)})")
            self._compute(f"in all of {', '.join(venues)}", "intersect the authors of the papers in each venue",
                          "author", name, [{"p": "authoredBy", "o": name}, {"p": "venue"}])
        return self._fact_answer(
            triples, f"{len(shared)} authors have indexed papers in all of {', '.join(venues)} "
                     f"[{', '.join(sorted({item.source_id for item in triples}))}]:\n" + "\n".join(lines))

    def _publication_year_extreme(self, scope: _Scope) -> FactResult:
        extreme = scope.plan.extreme
        if not extreme or not scope.years:
            return None
        year = (min if extreme == "earliest" else max)(int(row["object"]) for row in scope.years)
        triples = [self._fact_triple(row) for row in scope.years if int(row["object"]) == year]
        names = [self.corpus.paper_title(item.subject) or item.subject for item in triples]
        self._compute(f"{extreme} year: {year}",
                      f"{'minimum' if extreme == 'earliest' else 'maximum'} over all publication years",
                      "value", str(year), [{"p": "year", "o": str(year)}])
        return self._fact_answer(triples, f"The {extreme} publication year is {year}: {', '.join(names)} "
                                          f"[{', '.join(item.source_id for item in triples)}].")

    def _author_paper_counts(self, scope: _Scope) -> FactResult:
        plan = scope.plan
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in scope.facts:
            if row["predicate"] == "author":
                grouped.setdefault(str(row["object"]), []).append(row)
        distinct = {name: self._distinct_works(rows) for name, rows in grouped.items()}
        if not distinct:
            return None
        if plan.rank == "top":
            maximum = max(len(rows) for rows in distinct.values())
            selected = {name: rows for name, rows in distinct.items() if len(rows) == maximum}
            header = (f"{', '.join(sorted(selected))} "
                      f"{'appears' if len(selected) == 1 else 'appear'} on the most distinct corpus papers ({maximum})")
        else:
            threshold = plan.min_papers or 1
            selected = {name: rows for name, rows in distinct.items() if len(rows) >= threshold}
            header = f"Authors on at least {threshold} distinct corpus papers"
        triples = [self._fact_triple(row) for rows in selected.values() for row in rows]
        if not triples:
            raise CorpusAbsent(f"No author appears on at least {plan.min_papers} distinct corpus papers.",
                               "metadata_facts", len(scope.facts))
        lines = [
            f"{name} ({len(rows)}): " + ", ".join(self.corpus.paper_title(r["subject"]) or r["subject"] for r in rows)
            for name, rows in sorted(selected.items())
        ]
        for name, rows in sorted(selected.items()):
            self._compute(f"{len(rows)} distinct papers",
                          "count distinct corpus papers per author (a preprint and its journal version count once)",
                          "author", name, [{"p": "authoredBy", "o": name}])
        return self._fact_answer(
            triples, f"{header} [{', '.join(sorted({item.source_id for item in triples}))}]:\n" + "\n".join(lines)
        )

    def _citation_aggregate(self, scope: _Scope, name: str) -> tuple[list[dict[str, Any]], dict[str, dict]]:
        top = scope.plan.rank == "top"
        aggregate = self.corpus.aggregate(name, top_k=1 if top else None,
                                          min_count=None if top else (scope.plan.min_papers or 1))
        chosen = [item for item in aggregate["items"] if item["count"] > 0]
        return chosen, {str(row["reference_id"]): row for row in self.corpus.references}

    def _paper_citation_counts(self, scope: _Scope) -> FactResult:
        chosen, by_reference = self._citation_aggregate(scope, "count_incoming_citations")
        if not chosen:
            raise CorpusAbsent("No indexed paper is cited by another indexed paper.",
                               "citation_index", len(self.corpus.references))
        triples, lines = [], []
        for item in chosen:
            triples += [self._citation_triple(by_reference[ref]) for ref in item["reference_ids"]]
            noun = "citing paper" if item["count"] == 1 else "citing papers"
            lines.append(f"{item['title'] or item['paper_id']} ({item['count']} {noun})")
            self._compute(f"{item['count']} {noun}", "count distinct in-corpus papers citing the paper",
                          "subject", self._iri(item["paper_id"]), [{"p": "cites", "o": self._iri(item["paper_id"])}])
        header = "Most cited by other indexed papers: " if scope.plan.rank == "top" else "In-corpus citations: "
        return self._fact_answer(triples, header + "; ".join(lines)
                                 + f" [{', '.join(sorted({t.source_id for t in triples}))}].")

    def _author_citation_counts(self, scope: _Scope) -> FactResult:
        chosen, by_reference = self._citation_aggregate(scope, "count_citations_by_author")
        if not chosen:
            raise CorpusAbsent("No author's papers are cited by another indexed paper.",
                               "citation_index", len(self.corpus.references))
        triples, lines = [], []
        for item in chosen:
            works = set(item["paper_ids"])
            triples += [self._fact_triple(row) for row in scope.facts
                        if row["predicate"] == "author" and str(row["object"]) == item["author"]
                        and str(row["subject"]) in works]
            triples += [self._citation_triple(by_reference[ref]) for ref in item["reference_ids"]]
            noun = "citing paper" if item["count"] == 1 else "citing papers"
            lines.append(f"{item['author']} ({item['count']} {noun})")
            self._compute(f"{item['count']} {noun}",
                          "count distinct in-corpus papers citing the author's distinct papers",
                          "author", item["author"],
                          [{"p": "authoredBy", "o": item["author"]},
                           {"p": "cites", "o_in": [self._iri(paper) for paper in works]}])
        header = ("Most cited in the corpus: " if scope.plan.rank == "top" else "In-corpus citations: ") + "; ".join(lines)
        return self._fact_answer(triples, f"{header} [{', '.join(sorted({t.source_id for t in triples}))}].")

    def _iri(self, paper: str) -> str:
        """The identifier the answer triples use for a paper (DBLP IRI when verified)."""
        return self.corpus.canonical_paper_iri(paper) or paper

    @staticmethod
    def _fact_triple(row: dict[str, Any]) -> ExtractedTriple:
        return ExtractedTriple(
            subject=str(row["subject"]),
            predicate={"author": "authoredBy"}.get(str(row["predicate"]), str(row["predicate"])),
            object=str(row["object"]),
            source_id=str(row["source_id"]),
            evidence=f"Corpus fact {row.get('fact_id', row['source_id'])}.",
        )

    @staticmethod
    def _citation_triple(row: dict[str, Any]) -> ExtractedTriple:
        return ExtractedTriple(
            subject=str(row["paper_id"]), predicate="cites",
            object=str(row["resolved_paper_id"]),
            source_id=str(row["reference_id"]),
            evidence=str(row.get("title") or row.get("raw") or row["reference_id"]),
        )

    @staticmethod
    def _fact_answer(
        triples: list[ExtractedTriple], answer: str
    ) -> tuple[AnswerResult, list[ExtractedTriple]]:
        predicate = triples[0].predicate
        return (
            AnswerResult(
                sparql=(
                    "PREFIX hrn: <https://harness.local/schema#>\n"
                    f"SELECT ?subject ?object WHERE {{ ?subject hrn:{predicate} ?object . }}"
                ),
                answer=answer,
                cited_source_ids=sorted({item.source_id for item in triples}),
            ),
            triples,
        )

    def _distinct_works(self, author_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """A preprint and its near-identical journal version count as one work (shared corpus rule)."""
        return self.corpus.distinct_works(author_rows)

    def _author_triples(self, paper_id: str) -> list[ExtractedTriple]:
        return [
            ExtractedTriple(
                subject=str(row["subject"]), predicate="authoredBy", object=str(row["object"]),
                source_id=str(row["source_id"]), evidence=f"Corpus fact {row.get('fact_id', row['source_id'])}.",
            )
            for row in self.corpus.seed_facts()
            if row.get("subject") == paper_id and row.get("predicate") == "author"
        ]

    def _plan(self, question: str) -> QueryPlan:
        preset, self.preset_plan = getattr(self, "preset_plan", None), None
        if preset is not None:
            return preset
        if self.planner is None:
            return QueryPlan.fallback("no query planner configured")
        return self.planner.plan(question, self.concept)



# Query-plan operation -> the method that answers it (operations: planner.OPERATIONS).
FACT_OPERATIONS = {
    "find_paper": FactAnswers._find_paper,
    "paper_authors": FactAnswers._paper_authors,
    "paper_year": FactAnswers._paper_year_or_venue,
    "paper_venue": FactAnswers._paper_year_or_venue,
    "citing_papers": FactAnswers._citing_papers,
    "citing_count": FactAnswers._citing_papers,
    "cited_papers": FactAnswers._cited_papers,
    "cites": FactAnswers._cites,
    "papers_in_year": FactAnswers._papers_in_year,
    "papers_with_title_term": FactAnswers._papers_with_title_term,
    "papers_in_venue": FactAnswers._papers_in_venue,
    "authors_in_venues": FactAnswers._authors_in_venues,
    "publication_year_extreme": FactAnswers._publication_year_extreme,
    "author_paper_counts": FactAnswers._author_paper_counts,
    "paper_citation_counts": FactAnswers._paper_citation_counts,
    "author_citation_counts": FactAnswers._author_citation_counts,
}
