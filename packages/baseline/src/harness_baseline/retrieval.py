"""Retrieval step: choose the passages a research run reads (BM25, seeds, coverage).

A mixin of ``Baseline20``: methods use the pipeline's state (``self``).

``HARNESS_RETRIEVAL_MODE``:

- ``lexical`` (default): boolean DSL candidates, then BM25 ranking with a relative threshold.
- ``hybrid``: the same candidates plus the top of ``CorpusStore.hybrid_rank`` (BM25 without the
  boolean filter, fused with dense embeddings when the corpus has a dense index), ranked by the
  fused order and cut at ``retrieve_k``. Measure it against lexical on the gold set before
  making it the default.
"""

from __future__ import annotations

import re
import time
from typing import Any

from harness_config import settings

from .planner import QueryPlan
from .state import (
    BaselineState,
    NoEvidenceFound,
    retrieval_dsl,
)

LIMITATION = re.compile(r"\blimitations?\b", re.IGNORECASE)
RELATIVE_THRESHOLD = 0.70        # lexical mode keeps passages scoring at least this share of the best


class RetrievalStep:
    def _retrieve(self, state: BaselineState) -> dict[str, Any]:
        """Candidates (search rounds, known passages), ranking, then the passage budget."""
        self.stage = "retrieval"
        question = state["question"]
        plan = state.get("plan") or QueryPlan.fallback("no plan")
        concept = state.get("concept") or {}
        # Papers named in the plan scope retrieval. Without a model plan, fall
        # back to title detection over the question text.
        mentioned = list(state.get("paper_ids") or []) or (
            self.corpus.mentioned_paper_ids(question) if plan.source == "fallback" else []
        )
        queries = self._search_queries(question, concept)
        found, rounds = self._search_rounds(queries, mentioned)
        reused_claims = self._add_known_passages(found, concept, mentioned, question)
        focus = self._focus(question, concept, mentioned)
        hybrid = settings().retrieval_mode == "hybrid"
        methods = self._add_hybrid_candidates(found, rounds, focus, mentioned) if hybrid else "lexical+semantic-rank"
        if not found:
            raise NoEvidenceFound("No evidence was found in the indexed full text.", {
                "coverage_complete": False, "queries": len(queries), "passages_inspected": 0,
                "papers_in_scope": len(mentioned) or len(self.corpus.papers),
                "retrieval_methods": methods, "parser_failures": 0,
            })
        rank_started = time.monotonic()
        limitations = plan.evidence_focus == "limitations"
        rank = self._rank_hybrid if hybrid else self._rank_lexical
        rows, ranked, focus = rank(found, focus, question, mentioned, limitations)
        rank_duration_ms = round((time.monotonic() - rank_started) * 1000, 1)
        self._keep_seeds(rows, found)
        rows, excluded_for_budget = self._apply_budget(rows, mentioned)
        if not rows:
            raise NoEvidenceFound("No evidence was found after the configured full-text search.", {
                "coverage_complete": False, "queries": len(queries), "passages_inspected": len(ranked),
                "papers_in_scope": len(mentioned) or len(self.corpus.papers),
                "retrieval_methods": methods, "parser_failures": 0,
            })
        paper_count = len({row["paper_id"] for row in rows})
        self._emit("Retrieving top passages",
                   f"{len(rows)} passages from {paper_count} papers across {len(rounds)} rounds")
        self._check("retrieval", "pass", f"{len(rows)} passages from {paper_count} papers")
        return {"passages": rows, "retrieval": {
            "queries": rounds, "stop_reason": "hybrid_top_k" if hybrid else "relative_bm25_relevance",
            "retrieval_mode": "hybrid" if hybrid else "lexical",
            "mentioned_paper_ids": mentioned,
            "plan": plan.model_dump(),
            "reused_claim_ids": [claim.id for claim in reused_claims],
            "agent_seed_passage_ids": self.seed_passage_ids,
            "candidates": len(found), "selected": len(rows),
            "max_passages": self.max_passages,
            "excluded_for_budget": excluded_for_budget,
            "relative_score_threshold": RELATIVE_THRESHOLD,
            "ranking": {
                "query": focus, "candidate_ids": list(found),
                "passage_ids": [str(row["passage_id"]) for row in ranked],
                "duration_ms": rank_duration_ms,
            },
        }}

    # ─── Candidates ──────────────────────────────────────────────────────────

    def _search_queries(self, question: str, concept: dict[str, Any]) -> list[str]:
        """The boolean DSL query, word-pair phrases (enhanced runs) and the reading's search terms."""
        queries = [retrieval_dsl(question)]
        if self.enhanced:
            terms = [t for t in re.findall(r"[A-Za-z][A-Za-z0-9_-]+", question) if len(t) >= 4]
            phrases = [f'"{" ".join(terms[i : i + 2])}"' for i in range(max(0, len(terms) - 1))]
            queries.extend(phrases[:4])
        for term in concept.get("search_terms") or []:
            words = re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", str(term))
            if words:
                queries.append(f'"{" ".join(words)}"' if len(words) > 1 else words[0])
        return queries

    def _search_rounds(self, queries: list[str], mentioned: list[str]
                       ) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
        found: dict[str, dict[str, Any]] = {}
        rounds: list[dict[str, Any]] = []
        for round_no, query in enumerate(queries, start=1):
            # Scope explicit paper questions before broad lexical expansion.
            # Inspect all matches, then use the existing BM25S index to select
            # evidence by relevance rather than by a fixed passage count.
            per_query_limit = max(1, len(self.corpus.passages))
            query_started = time.monotonic()
            scope = {"paper_ids": mentioned} if mentioned else None
            hits = self.corpus.search(query, scope=scope, limit=per_query_limit)["items"]
            before = len(found)
            found.update({row["passage_id"]: row for row in hits})
            rounds.append({
                "round": round_no, "query": query, "limit": per_query_limit,
                "returned": len(hits), "new": len(found) - before,
                "passage_ids": [str(row["passage_id"]) for row in hits],
                "duration_ms": round((time.monotonic() - query_started) * 1000, 1),
            })
        return found, rounds

    def _add_known_passages(self, found: dict[str, dict[str, Any]], concept: dict[str, Any],
                            mentioned: list[str], question: str) -> list[Any]:
        """The reading's grounding passages, reused claims' sources and agent seeds become candidates
        (ranked like any other, not forced in). Returns the reused claims."""
        # When the question names papers, only their passages qualify.
        for source_id in concept.get("sample_sources") or []:
            passage = self.corpus.by_passage.get(str(source_id))
            if passage is not None and (not mentioned or str(passage["paper_id"]) in mentioned):
                found.setdefault(str(source_id), passage)
        reused_claims = self.claim_lookup(question) if self.claim_lookup else []
        known = [source_id for claim in reused_claims for source_id in claim.source_ids]
        for source_id in [*known, *self.seed_passage_ids]:
            passage = self.corpus.by_passage.get(source_id)
            if passage is not None:
                found.setdefault(source_id, passage)
        return reused_claims

    def _focus(self, question: str, concept: dict[str, Any], mentioned: list[str]) -> str:
        """The ranking query: the question (and reading) without the named papers' own names."""
        focus = question
        if concept:
            focus = f"{question} {concept.get('label', '')} {concept.get('gloss', '')}"
        for paper_id in mentioned:
            title = self.corpus.paper_title(paper_id) or ""
            prefix = " ".join(title.split()[:3])
            if prefix:
                focus = re.sub(re.escape(prefix), " ", focus, flags=re.IGNORECASE)
            brand = re.split(r"[^A-Za-z0-9]+", title, maxsplit=1)[0]
            if brand and (brand.isupper() or any(char.isdigit() for char in brand)
                          or any(char.isupper() for char in brand[1:])):
                focus = re.sub(rf"(?i)\b{re.escape(brand)}\b", " ", focus)
        return focus if re.search(r"[A-Za-z]{3,}", focus) else question

    def _add_hybrid_candidates(self, found: dict[str, dict[str, Any]], rounds: list[dict[str, Any]],
                               focus: str, mentioned: list[str]) -> str:
        """Natural-language candidates: no boolean filter, dense embeddings when indexed."""
        query_started = time.monotonic()
        depth = max(self.retrieve_k, 3 * (self.max_passages or self.retrieve_k))
        scope = {"paper_ids": mentioned} if mentioned else None
        extra = self.corpus.hybrid_rank(focus, scope=scope, limit=depth)
        before = len(found)
        for row in extra["items"]:
            found.setdefault(str(row["passage_id"]), self.corpus.by_passage[str(row["passage_id"])])
        rounds.append({
            "round": len(rounds) + 1, "query": f"hybrid: {focus}", "limit": depth,
            "returned": len(extra["items"]), "new": len(found) - before,
            "passage_ids": [str(row["passage_id"]) for row in extra["items"]],
            "retrieval": extra["retrieval"],
            "duration_ms": round((time.monotonic() - query_started) * 1000, 1),
        })
        return f"lexical+hybrid({extra['retrieval']})"

    # ─── Ranking: (selected rows, full ranking, query used) ──────────────────

    def _rank_hybrid(self, found: dict[str, dict[str, Any]], focus: str, question: str,
                     mentioned: list[str], limitations: bool) -> tuple[list[dict], list[dict], str]:
        def rank(query: str) -> list[dict[str, Any]]:
            return self.corpus.hybrid_rank(query, passage_ids=list(found), limit=max(1, len(found)))["items"]

        ranked = rank(focus)
        if focus != question and not ranked:
            # As in lexical mode: removing the paper's own name left nothing that ranks.
            focus = question
            ranked = rank(focus)
        top = ranked[: self.retrieve_k]
        top_ids = {str(row["passage_id"]) for row in top}
        if limitations:
            top += [row for row in ranked[self.retrieve_k: 2 * self.retrieve_k]
                    if str(row["passage_id"]) not in top_ids and LIMITATION.search(row["text"])]
        rows = [{key: value for key, value in row.items()
                 if key not in {"score", "bm25_rank", "dense_rank"}} for row in top]
        return rows, ranked, focus

    def _rank_lexical(self, found: dict[str, dict[str, Any]], focus: str, question: str,
                      mentioned: list[str], limitations: bool) -> tuple[list[dict], list[dict], str]:
        def rank(query: str) -> list[dict[str, Any]]:
            return self.corpus.rank_evidence(query, passage_ids=list(found), limit=max(1, len(found)))["items"]

        ranked = rank(focus)
        if focus != question and not any(float(row["score"]) > 0 for row in ranked):
            # Removing the paper's own name left nothing that scores (e.g. the
            # question was mostly the title): rank by the full question instead.
            focus = question
            ranked = rank(focus)

        def group(row: dict[str, Any]) -> str:
            return str(row["paper_id"]) if mentioned else "all"

        best: dict[str, float] = {}
        for row in ranked:
            best[group(row)] = max(best.get(group(row), 0.0), float(row["score"]))
        rows = [
            {key: value for key, value in row.items() if key != "score"}
            for row in ranked
            if row["score"] > 0
            and (row["score"] >= RELATIVE_THRESHOLD * best[group(row)]
                 or limitations and LIMITATION.search(row["text"]))
        ]
        if not self.enhanced:
            rows = rows[: self.retrieve_k]
        return rows, ranked, focus

    # ─── Selection ───────────────────────────────────────────────────────────

    def _keep_seeds(self, rows: list[dict[str, Any]], found: dict[str, dict[str, Any]]) -> None:
        """Agent-selected passages were already observed through MCP: keep them even when ranking the
        question gives them a low score, or the agent's search could not help extraction and answer."""
        selected_ids = {str(row["passage_id"]) for row in rows}
        for source_id in self.seed_passage_ids:
            passage = found.get(source_id)
            if passage is not None and source_id not in selected_ids:
                rows.append(passage)
                selected_ids.add(source_id)

    def _apply_budget(self, rows: list[dict[str, Any]], mentioned: list[str]) -> tuple[list[dict[str, Any]], int]:
        """At most ``max_passages``: named papers' share, then seeds, one per paper, then by rank."""
        if self.max_passages is None or len(rows) <= self.max_passages:
            return rows, 0
        limit = self.max_passages
        selected: list[dict[str, Any]] = []
        used: set[str] = set()

        def take(row: dict[str, Any]) -> None:
            selected.append(row)
            used.add(str(row["passage_id"]))

        # A question that names several papers (a comparison) gets an even share for each named
        # paper first: half the budget, split between them, seeds of that paper before its best-
        # ranked passages. Otherwise seeds of one paper could fill the budget and drop the other.
        present = {str(row["paper_id"]) for row in rows}
        named = [paper for paper in dict.fromkeys(str(value) for value in mentioned) if paper in present]
        if len(named) >= 2:
            share = max(1, limit // (2 * len(named)))
            seed_rank = {source_id: index for index, source_id in enumerate(self.seed_passage_ids)}
            for paper_id in named:
                own = [row for row in rows if str(row["paper_id"]) == paper_id]
                own.sort(key=lambda row: (str(row["passage_id"]) not in seed_rank,
                                          seed_rank.get(str(row["passage_id"]), 0)))
                for row in own[:share]:
                    if len(selected) < limit:
                        take(row)
        by_id = {str(row["passage_id"]): row for row in rows}
        for source_id in self.seed_passage_ids:
            if source_id in by_id and source_id not in used and len(selected) < limit:
                take(by_id[source_id])
        represented = {str(row["paper_id"]) for row in selected}
        for row in rows:
            if len(selected) >= limit:
                break
            if str(row["passage_id"]) not in used and str(row["paper_id"]) not in represented:
                take(row)
                represented.add(str(row["paper_id"]))
        for row in rows:
            if len(selected) >= limit:
                break
            if str(row["passage_id"]) not in used:
                take(row)
        excluded = len(rows) - len(selected)
        self._emit("Bounding extraction scope",
                   f"{len(selected)} passages selected for this run; {excluded} lower-priority passages deferred")
        return selected, excluded
