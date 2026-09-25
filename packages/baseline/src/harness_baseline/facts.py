"""Fact operations: answer metadata and citation questions from verified DBLP facts.

A mixin of ``Baseline20``: methods use the pipeline's state (``self``).
"""

from __future__ import annotations

from typing import Any

from harness_extraction import (
    AnswerResult,
    ExtractedTriple,
)

from .planner import PaperResolver, QueryPlan
from .state import (
    CorpusAbsent,
)


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

        Returns None when the facts needed for the operation are not indexed,
        so the caller can fall back to full-text research. Raises CorpusAbsent
        when the operation completed and established absence on its surface.
        """
        facts = self.corpus.seed_facts()
        op = plan.operation
        target = self._title_row(papers[0]) if papers else None
        # Conclusions computed below (counts, extremes) with the facts they rest on.
        self._computed: list[dict[str, Any]] = []

        if op == "find_paper" and target:
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

        if op == "paper_authors" and target:
            triples = self._author_triples(papers[0])
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

        if op in {"paper_year", "paper_venue"} and target:
            predicate = "year" if op == "paper_year" else "venue"
            rows = [row for row in facts if row["subject"] == papers[0] and row["predicate"] == predicate]
            if not rows:
                return None
            triples = [self._fact_triple(row) for row in rows]
            return self._fact_answer(
                triples, f"{target['object']} was published in {rows[0]['object']} [{rows[0]['source_id']}]."
            )

        if op in {"citing_papers", "citing_count"} and target:
            rows = [row for row in self.corpus.references if row.get("resolved_paper_id") == papers[0]]
            if not rows:
                raise CorpusAbsent(
                    f"No in-corpus evidence found that a paper cites {target['object']}. "
                    "This does not establish that no external citation exists.",
                    "citation_index", len(self.corpus.references),
                )
            triples = [self._citation_triple(row) for row in rows]
            sources = ", ".join(item.source_id for item in triples)
            if op == "citing_count":
                count = len({item.subject for item in triples})
                self._computed.append({
                    "label": f"{count} citing papers", "operation": "count distinct in-corpus papers citing it",
                    "about": {"role": "subject", "value": self._iri(papers[0])},
                    "support": [{"p": "cites", "o": self._iri(papers[0])}],
                })
                return self._fact_answer(triples, f"{count} indexed papers cite {target['object']} [{sources}].")
            names = sorted({self.corpus.paper_title(item.subject) or item.subject for item in triples})
            return self._fact_answer(triples, f"{', '.join(names)} cite {target['object']} [{sources}].")

        if op == "cited_papers" and target:
            rows = [row for row in self.corpus.references
                    if row["paper_id"] == papers[0] and row.get("resolved_paper_id")]
            if not rows:
                raise CorpusAbsent(f"No resolved in-corpus citations found for {target['object']}.",
                                   "citation_index", len(self.corpus.references))
            triples = [self._citation_triple(row) for row in rows]
            names = [self.corpus.paper_title(item.object) or item.object for item in triples]
            return self._fact_answer(triples, f"{target['object']} cites these indexed papers: {', '.join(names)} [{', '.join(item.source_id for item in triples)}].")

        if op == "cites" and len(papers) >= 2 and target:
            cited = self._title_row(papers[1])
            rows = [row for row in self.corpus.references
                    if row["paper_id"] == papers[0] and row.get("resolved_paper_id") == papers[1]]
            if not rows:
                raise CorpusAbsent(f"No resolved in-corpus citation from {target['object']} to {cited['object']}.",
                                   "citation_index", len(self.corpus.references))
            triples = [self._citation_triple(row) for row in rows]
            return self._fact_answer(triples, f"Yes. {target['object']} cites {cited['object']} [{', '.join(item.source_id for item in triples)}].")

        years = [row for row in facts if row["predicate"] == "year" and str(row["object"]).isdigit()]
        if op == "papers_in_year" and plan.year is not None:
            rows = [row for row in years if int(row["object"]) == plan.year]
            if not rows:
                raise CorpusAbsent(f"No indexed paper has publication year {plan.year}.",
                                   "metadata_facts", len(years))
            triples = [self._fact_triple(row) for row in rows]
            names = [self.corpus.paper_title(item.subject) or item.subject for item in triples]
            self._computed.append({
                "label": f"{len({item.subject for item in triples})} papers in {plan.year}",
                "operation": "count papers whose publication year matches",
                "about": {"role": "value", "value": str(plan.year)},
                "support": [{"p": "year", "o": str(plan.year)}],
            })
            return self._fact_answer(triples, f"{len({item.subject for item in triples})} corpus papers were published in {plan.year}: {', '.join(names)} [{', '.join(item.source_id for item in triples)}].")

        if op == "publication_year_extreme" and plan.extreme and years:
            pick = min if plan.extreme == "earliest" else max
            year = pick(int(row["object"]) for row in years)
            rows = [row for row in years if int(row["object"]) == year]
            triples = [self._fact_triple(row) for row in rows]
            names = [self.corpus.paper_title(item.subject) or item.subject for item in triples]
            self._computed.append({
                "label": f"{plan.extreme} year: {year}",
                "operation": f"{'minimum' if plan.extreme == 'earliest' else 'maximum'} over all publication years",
                "about": {"role": "value", "value": str(year)},
                "support": [{"p": "year", "o": str(year)}],
            })
            return self._fact_answer(triples, f"The {plan.extreme} publication year is {year}: {', '.join(names)} [{', '.join(item.source_id for item in triples)}].")

        if op == "author_paper_counts":
            grouped: dict[str, list[dict[str, Any]]] = {}
            for row in facts:
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
                                   "metadata_facts", len(facts))
            lines = [
                f"{name} ({len(rows)}): " + ", ".join(self.corpus.paper_title(r["subject"]) or r["subject"] for r in rows)
                for name, rows in sorted(selected.items())
            ]
            for name, rows in sorted(selected.items()):
                self._computed.append({
                    "label": f"{len(rows)} distinct papers",
                    "operation": "count distinct corpus papers per author (a preprint and its journal version count once)",
                    "about": {"role": "author", "value": name},
                    "support": [{"p": "authoredBy", "o": name}],
                })
            return self._fact_answer(
                triples, f"{header} [{', '.join(sorted({item.source_id for item in triples}))}]:\n" + "\n".join(lines)
            )

        if op == "author_citation_counts":
            aggregate = self.corpus.aggregate(
                "count_citations_by_author", top_k=1 if plan.rank == "top" else None,
                min_count=None if plan.rank == "top" else (plan.min_papers or 1),
            )
            chosen = [item for item in aggregate["items"] if item["count"] > 0]
            if not chosen:
                raise CorpusAbsent("No author's papers are cited by another indexed paper.",
                                   "citation_index", len(self.corpus.references))
            by_reference = {str(row["reference_id"]): row for row in self.corpus.references}
            triples, lines = [], []
            for item in chosen:
                works = set(item["paper_ids"])
                triples += [self._fact_triple(row) for row in facts
                            if row["predicate"] == "author" and str(row["object"]) == item["author"]
                            and str(row["subject"]) in works]
                triples += [self._citation_triple(by_reference[ref]) for ref in item["reference_ids"]]
                noun = "citing paper" if item["count"] == 1 else "citing papers"
                lines.append(f"{item['author']} ({item['count']} {noun})")
                self._computed.append({
                    "label": f"{item['count']} {noun}",
                    "operation": "count distinct in-corpus papers citing the author's distinct papers",
                    "about": {"role": "author", "value": item["author"]},
                    "support": [{"p": "authoredBy", "o": item["author"]},
                                {"p": "cites", "o_in": [self._iri(paper) for paper in works]}],
                })
            header = ("Most cited in the corpus: " if plan.rank == "top" else "In-corpus citations: ") + "; ".join(lines)
            return self._fact_answer(triples, f"{header} [{', '.join(sorted({t.source_id for t in triples}))}].")
        return None

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
        if self.planner is None:
            return QueryPlan.fallback("no query planner configured")
        return self.planner.plan(question, self.concept)

