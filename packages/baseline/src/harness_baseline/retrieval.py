"""Retrieval step: choose the passages a research run reads (BM25, seeds, coverage).

A mixin of ``Baseline20``: methods use the pipeline's state (``self``).
"""

from __future__ import annotations

import re
import time
from typing import Any

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
        if not found:
            raise NoEvidenceFound("No evidence was found in the indexed full text.", {
                "coverage_complete": False, "queries": len(queries), "passages_inspected": 0,
                "papers_in_scope": len(mentioned_papers) or len(self.corpus.papers),
                "retrieval_methods": "lexical+semantic-rank", "parser_failures": 0,
            })
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
        rank_started = time.monotonic()
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
        rank_duration_ms = round((time.monotonic() - rank_started) * 1000, 1)
        best_by_paper: dict[str, float] = {}
        for row in ranked:
            key = str(row["paper_id"]) if mentioned_papers else "all"
            best_by_paper[key] = max(best_by_paper.get(key, 0.0), float(row["score"]))
        relative_threshold = 0.70
        limitation_question = plan.evidence_focus == "limitations"
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
                "retrieval_methods": "lexical+semantic-rank", "parser_failures": 0,
            })
        paper_count = len({row["paper_id"] for row in rows})
        self._emit("Retrieving top passages",
                   f"{len(rows)} passages from {paper_count} papers across {len(rounds)} rounds")
        self._check("retrieval", "pass", f"{len(rows)} passages from {paper_count} papers")
        return {"passages": rows, "retrieval": {
            "queries": rounds, "stop_reason": "relative_bm25_relevance",
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

