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


class RetrievalStep:
    def _retrieve(self, state: BaselineState) -> dict[str, Any]:
        self.stage = "retrieval"
        question = state["question"]
        plan = state.get("plan") or QueryPlan.fallback("no plan")
        # Papers named in the plan scope retrieval. Without a model plan, fall
        # back to title detection over the question text.
        mentioned_papers = list(state.get("paper_ids") or []) or (
            self.corpus.mentioned_paper_ids(question) if plan.source == "fallback" else []
        )
        dsl = retrieval_dsl(question)
        queries = [dsl]
        if self.enhanced:
            terms = [t for t in re.findall(r"[A-Za-z][A-Za-z0-9_-]+", question) if len(t) >= 4]
            phrases = [f'"{" ".join(terms[i : i + 2])}"' for i in range(max(0, len(terms) - 1))]
            queries.extend(phrases[:4])
        concept = state.get("concept") or {}
        for term in concept.get("search_terms") or []:
            words = re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", str(term))
            if words:
                queries.append(f'"{" ".join(words)}"' if len(words) > 1 else words[0])
        found: dict[str, dict[str, Any]] = {}
        rounds: list[dict[str, Any]] = []
        for round_no, query in enumerate(queries, start=1):
            # Scope explicit paper questions before broad lexical expansion.
            # Inspect all matches, then use the existing BM25S index to select
            # evidence by relevance rather than by a fixed passage count.
            per_query_limit = max(1, len(self.corpus.passages))
            query_started = time.monotonic()
            scope = {"paper_ids": mentioned_papers} if mentioned_papers else None
            hits = self.corpus.search(query, scope=scope, limit=per_query_limit)["items"]
            before = len(found)
            found.update({row["passage_id"]: row for row in hits})
            rounds.append({
                "round": round_no, "query": query, "limit": per_query_limit,
                "returned": len(hits), "new": len(found) - before,
                "passage_ids": [str(row["passage_id"]) for row in hits],
                "duration_ms": round((time.monotonic() - query_started) * 1000, 1),
            })
        # Passages that grounded the chosen reading are candidates too (they are ranked like
        # any other candidate, not forced in).
        # When the question names papers, only their passages qualify.
        for source_id in concept.get("sample_sources") or []:
            passage = self.corpus.by_passage.get(str(source_id))
            if passage is not None and (not mentioned_papers or str(passage["paper_id"]) in mentioned_papers):
                found.setdefault(str(source_id), passage)
        reused_claims = self.claim_lookup(question) if self.claim_lookup else []
        for claim in reused_claims:
            for source_id in claim.source_ids:
                passage = self.corpus.by_passage.get(source_id)
                if passage is not None:
                    found.setdefault(source_id, passage)
        for source_id in self.seed_passage_ids:
            passage = self.corpus.by_passage.get(source_id)
            if passage is not None:
                found.setdefault(source_id, passage)
        focus = question
        if concept:
            focus = f"{question} {concept.get('label', '')} {concept.get('gloss', '')}"
        for paper_id in mentioned_papers:
            title = self.corpus.paper_title(paper_id) or ""
            prefix = " ".join(title.split()[:3])
            if prefix:
                focus = re.sub(re.escape(prefix), " ", focus, flags=re.IGNORECASE)
            brand = re.split(r"[^A-Za-z0-9]+", title, maxsplit=1)[0]
            if brand and (brand.isupper() or any(char.isdigit() for char in brand) or any(char.isupper() for char in brand[1:])):
                focus = re.sub(rf"(?i)\b{re.escape(brand)}\b", " ", focus)
        if not re.search(r"[A-Za-z]{3,}", focus):
            focus = question
        hybrid = settings().retrieval_mode == "hybrid"
        methods = "lexical+semantic-rank"
        if hybrid:
            # Natural-language candidates: no boolean filter, dense embeddings when indexed.
            query_started = time.monotonic()
            depth = max(self.retrieve_k, 3 * (self.max_passages or self.retrieve_k))
            scope = {"paper_ids": mentioned_papers} if mentioned_papers else None
            extra = self.corpus.hybrid_rank(focus, scope=scope, limit=depth)
            before = len(found)
            for row in extra["items"]:
                found.setdefault(str(row["passage_id"]), self.corpus.by_passage[str(row["passage_id"])])
            methods = f"lexical+hybrid({extra['retrieval']})"
            rounds.append({
                "round": len(rounds) + 1, "query": f"hybrid: {focus}", "limit": depth,
                "returned": len(extra["items"]), "new": len(found) - before,
                "passage_ids": [str(row["passage_id"]) for row in extra["items"]],
                "retrieval": extra["retrieval"],
                "duration_ms": round((time.monotonic() - query_started) * 1000, 1),
            })
        if not found:
            raise NoEvidenceFound("No evidence was found in the indexed full text.", {
                "coverage_complete": False, "queries": len(queries), "passages_inspected": 0,
                "papers_in_scope": len(mentioned_papers) or len(self.corpus.papers),
                "retrieval_methods": methods, "parser_failures": 0,
            })
        rank_started = time.monotonic()
        limitation_question = plan.evidence_focus == "limitations"
        relative_threshold = 0.70
        if hybrid:
            ranked = self.corpus.hybrid_rank(focus, passage_ids=list(found), limit=max(1, len(found)))["items"]
            if focus != question and not ranked:
                # As in lexical mode: removing the paper's own name left nothing that ranks.
                focus = question
                ranked = self.corpus.hybrid_rank(focus, passage_ids=list(found), limit=max(1, len(found)))["items"]
            top = ranked[: self.retrieve_k]
            top_ids = {str(row["passage_id"]) for row in top}
            if limitation_question:
                top += [row for row in ranked[self.retrieve_k: 2 * self.retrieve_k]
                        if str(row["passage_id"]) not in top_ids
                        and re.search(r"\blimitations?\b", row["text"], re.IGNORECASE)]
            rows = [{key: value for key, value in row.items()
                     if key not in {"score", "bm25_rank", "dense_rank"}} for row in top]
        else:
            ranked = self.corpus.rank_evidence(
                focus, passage_ids=list(found), limit=max(1, len(found))
            )["items"]
            if focus != question and not any(float(row["score"]) > 0 for row in ranked):
                # Removing the paper's own name left nothing that scores (e.g. the
                # question was mostly the title): rank by the full question instead.
                focus = question
                ranked = self.corpus.rank_evidence(
                    focus, passage_ids=list(found), limit=max(1, len(found))
                )["items"]
            best_by_paper: dict[str, float] = {}
            for row in ranked:
                key = str(row["paper_id"]) if mentioned_papers else "all"
                best_by_paper[key] = max(best_by_paper.get(key, 0.0), float(row["score"]))
            rows = [
                {key: value for key, value in row.items() if key != "score"}
                for row in ranked
                if row["score"] > 0
                and (
                    row["score"] >= relative_threshold * best_by_paper[
                        str(row["paper_id"]) if mentioned_papers else "all"
                    ]
                    or limitation_question and re.search(r"\blimitations?\b", row["text"], re.IGNORECASE)
                )
            ]
            if not self.enhanced:
                rows = rows[: self.retrieve_k]
        rank_duration_ms = round((time.monotonic() - rank_started) * 1000, 1)
        # Agent-selected passages have already been observed through MCP.
        # Keep them even when lexical ranking of the original question gives
        # them a low score; otherwise the early MCP search cannot help the
        # extraction and final answer stages.
        selected_ids = {str(row["passage_id"]) for row in rows}
        for source_id in self.seed_passage_ids:
            passage = found.get(source_id)
            if passage is not None and source_id not in selected_ids:
                rows.append(passage)
                selected_ids.add(source_id)
        excluded_for_budget = 0
        if self.max_passages is not None and len(rows) > self.max_passages:
            before_cap = len(rows)
            by_id = {str(row["passage_id"]): row for row in rows}
            selected: list[dict[str, Any]] = []
            used: set[str] = set()
            # A question that names several papers (a comparison) gets an even share for each named
            # paper first: half the budget, split between them, seeds of that paper before its best-
            # ranked passages. Otherwise seeds of one paper could fill the budget and drop the other.
            present = {str(row["paper_id"]) for row in rows}
            named = [paper for paper in dict.fromkeys(str(value) for value in mentioned_papers) if paper in present]
            if len(named) >= 2:
                share = max(1, self.max_passages // (2 * len(named)))
                seed_rank = {source_id: index for index, source_id in enumerate(self.seed_passage_ids)}
                for paper_id in named:
                    own = [row for row in rows if str(row["paper_id"]) == paper_id]
                    own.sort(key=lambda row: (str(row["passage_id"]) not in seed_rank,
                                              seed_rank.get(str(row["passage_id"]), 0)))
                    for row in own[:share]:
                        if len(selected) < self.max_passages:
                            selected.append(row)
                            used.add(str(row["passage_id"]))
            for source_id in self.seed_passage_ids:
                if source_id in by_id and source_id not in used and len(selected) < self.max_passages:
                    selected.append(by_id[source_id])
                    used.add(source_id)
            represented = {str(row["paper_id"]) for row in selected}
            for row in rows:
                source_id = str(row["passage_id"])
                paper_id = str(row["paper_id"])
                if len(selected) >= self.max_passages:
                    break
                if source_id not in used and paper_id not in represented:
                    selected.append(row)
                    used.add(source_id)
                    represented.add(paper_id)
            for row in rows:
                source_id = str(row["passage_id"])
                if len(selected) >= self.max_passages:
                    break
                if source_id not in used:
                    selected.append(row)
                    used.add(source_id)
            rows = selected
            excluded_for_budget = before_cap - len(rows)
            self._emit(
                "Bounding extraction scope",
                f"{len(rows)} passages selected for this run; {excluded_for_budget} lower-priority passages deferred",
            )
        if not rows:
            raise NoEvidenceFound("No evidence was found after the configured full-text search.", {
                "coverage_complete": False, "queries": len(queries), "passages_inspected": len(ranked),
                "papers_in_scope": len(mentioned_papers) or len(self.corpus.papers),
                "retrieval_methods": methods, "parser_failures": 0,
            })
        paper_count = len({row["paper_id"] for row in rows})
        self._emit("Retrieving top passages",
                   f"{len(rows)} passages from {paper_count} papers across {len(rounds)} rounds")
        self._check("retrieval", "pass", f"{len(rows)} passages from {paper_count} papers")
        return {"passages": rows, "retrieval": {
            "queries": rounds, "stop_reason": "hybrid_top_k" if hybrid else "relative_bm25_relevance",
            "retrieval_mode": "hybrid" if hybrid else "lexical",
            "mentioned_paper_ids": mentioned_papers,
            "plan": plan.model_dump(),
            "reused_claim_ids": [claim.id for claim in reused_claims],
            "agent_seed_passage_ids": self.seed_passage_ids,
            "candidates": len(found), "selected": len(rows),
            "max_passages": self.max_passages,
            "excluded_for_budget": excluded_for_budget,
            "relative_score_threshold": relative_threshold,
            "ranking": {
                "query": focus, "candidate_ids": list(found),
                "passage_ids": [str(row["passage_id"]) for row in ranked],
                "duration_ms": rank_duration_ms,
            },
        }}

