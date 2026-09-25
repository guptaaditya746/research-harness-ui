"""Read-only corpus tools over ingest artifacts, a BM25S index and an optional dense index.

Retrieval roles:

- ``grep``: exact regular expressions.
- ``search``: exact lexical/boolean DSL (every term must appear; phrases, OR, NOT, NEAR/n, fields).
- ``hybrid_rank``: ranked natural-language retrieval: BM25 over all passages in scope (no boolean
  filter) fused with dense embeddings (reciprocal rank fusion) when ``index/dense`` exists, and an
  optional reranker (``HARNESS_MODEL_RERANK``). The agent's ``search_evidence`` uses it.
- ``rank_evidence``: BM25 scores for a given set of passages.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

import bm25s
import httpx
import numpy as np
import regex as safe_regex
from harness_config import settings

from .dense import QUERY_INSTRUCTION, DenseIndex, embed_texts, rrf

LOGGER = logging.getLogger(__name__)
FUSION_DEPTH = 200     # candidates taken from each ranking before fusion
RERANK_DEPTH = 30      # fused candidates sent to the optional reranker

PATTERN_MAX_LENGTH = 300
PATTERN_BUDGET_S = 2.0


class BoundedPattern:
    """A model-written regex with a total time budget per tool call.

    Tool callers are language models; a pathological pattern (catastrophic
    backtracking) would otherwise block the single-threaded tool server for
    every later run. The ``regex`` engine supports per-search timeouts.
    """

    def __init__(self, pattern: str, budget_s: float = PATTERN_BUDGET_S) -> None:
        if len(pattern) > PATTERN_MAX_LENGTH:
            raise ValueError(f"pattern is longer than {PATTERN_MAX_LENGTH} characters; simplify it")
        try:
            self.compiled = safe_regex.compile(pattern, safe_regex.IGNORECASE)
        except safe_regex.error as exc:
            raise ValueError(f"invalid pattern: {exc}") from exc
        self.deadline = time.monotonic() + budget_s

    def search(self, text: str):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError("pattern search exceeded its time budget; use a simpler pattern")
        try:
            return self.compiled.search(text, timeout=remaining)
        except TimeoutError as exc:
            raise ValueError("pattern search exceeded its time budget; use a simpler pattern") from exc

def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


class UnknownIdentifier(ValueError):
    """An identifier the corpus does not know; the message says what is accepted."""


def paged(rows: list[dict], offset: int = 0, limit: int = 10) -> dict:
    if offset < 0 or limit < 1:
        raise ValueError("offset must be nonnegative and limit must be positive")
    return {
        "total": len(rows),
        "offset": offset,
        "limit": limit,
        "items": rows[offset : offset + limit],
    }


def in_scope(paper: dict, scope: dict | None) -> bool:
    scope = scope or {}
    ids = scope.get("paper_ids")
    if ids and paper["paper_id"] not in ids:
        return False
    years = scope.get("year_range")
    return not years or paper["year"] is not None and years[0] <= paper["year"] <= years[1]


def lex(dsl: str) -> list[str]:
    tokens = re.findall(
        r'"[^"\n]+"|\(|\)|\bNEAR/\d+\b|\bAND\b|\bOR\b|\bNOT\b|[^\s()]+', dsl, re.IGNORECASE
    )
    if not tokens:
        raise ValueError("empty search")
    return tokens


def matches(dsl: str, text: str) -> bool:
    """Evaluate phrase, boolean, and proximity terms against one passage."""
    tokens = lex(dsl)
    words = re.findall(r"\w+", text.casefold())
    haystack = " ".join(words)
    pos = 0

    def term(value: str) -> bool:
        phrase = " ".join(re.findall(r"\w+", value.casefold()))
        return bool(phrase) and phrase in haystack

    def near(left: str, right: str, distance: int) -> bool:
        a = re.findall(r"\w+", left.casefold())
        b = re.findall(r"\w+", right.casefold())
        if not a or not b:
            return False
        left_pos = [i for i in range(len(words)) if words[i : i + len(a)] == a]
        right_pos = [i for i in range(len(words)) if words[i : i + len(b)] == b]
        return any(0 <= j - (i + len(a)) <= distance for i in left_pos for j in right_pos)

    def atom() -> bool:
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError("incomplete search expression")
        token = tokens[pos]
        pos += 1
        if token.upper() == "NOT":
            return not atom()
        if token == "(":
            value = expression()
            if pos >= len(tokens) or tokens[pos] != ")":
                raise ValueError("unclosed parenthesis")
            pos += 1
            return value
        if token == ")":
            raise ValueError("unexpected parenthesis")
        if pos + 1 < len(tokens) and re.fullmatch(r"NEAR/\d+", tokens[pos], re.IGNORECASE):
            distance = int(tokens[pos].split("/")[1])
            other = tokens[pos + 1]
            pos += 2
            return near(token, other, distance)
        return term(token)

    def conjunction() -> bool:
        nonlocal pos
        value = atom()
        while pos < len(tokens) and tokens[pos] != ")" and tokens[pos].upper() != "OR":
            if tokens[pos].upper() == "AND":
                pos += 1
            other = atom()
            value = value and other
        return value

    def expression() -> bool:
        nonlocal pos
        value = conjunction()
        while pos < len(tokens) and tokens[pos].upper() == "OR":
            pos += 1
            other = conjunction()
            value = value or other
        return value

    result = expression()
    if pos != len(tokens):
        raise ValueError("unexpected search token")
    return result


class CorpusStore:
    def __init__(self, root: Path):
        self.root = root
        self.papers = read_jsonl(root / "papers.jsonl")
        self.passages = read_jsonl(root / "passages.jsonl")
        self.references = read_jsonl(root / "references.jsonl")
        self.by_reference = {r["reference_id"]: r for r in self.references}
        self.fact_rows = read_jsonl(root / "facts.jsonl")
        self.by_paper = {p["paper_id"]: p for p in self.papers}
        self.by_dblp = {
            str(p["dblp_iri"]): p["paper_id"]
            for p in self.papers
            if p.get("dblp_iri")
        }
        self.by_passage = {p["passage_id"]: p for p in self.passages}
        index_path = root / "index" / "bm25"
        self.bm25 = (
            bm25s.BM25.load(str(index_path), load_corpus=True)
            if (index_path / "params.index.json").exists()
            else None
        )
        self.dense = DenseIndex(root, self.by_passage)

    def seed_facts(self) -> list[dict]:
        """Return deterministic, corpus-wide facts suitable for graph seeding."""
        return [
            row
            for row in self.fact_rows
            if row.get("predicate") != "cites" or row.get("in_corpus") is True
        ]

    # ─── Identifiers ──────────────────────────────────────────────────────────

    def paper_key(self, value: str) -> str:
        """Normalise any accepted paper identifier to the local key (``p23``).

        Accepted: the local key (``p23``), a verified DBLP record IRI
        (``https://dblp.org/rec/...``), a harness resource IRI
        (``https://harness.local/resource/p23``), or a source/fact id that starts
        with the key (``p23-metadata``, ``p23-abstract``). Anything else raises
        UnknownIdentifier with the accepted formats, so a wrong id is never
        mistaken for an empty corpus result.
        """
        raw = str(value or "").strip()
        if raw in self.by_paper:
            return raw
        if raw in self.by_dblp:
            return self.by_dblp[raw]
        if raw.startswith("https://harness.local/resource/"):
            tail = raw.rsplit("/", 1)[-1]
            if tail in self.by_paper:
                return tail
        prefix = re.match(r"(p\d+)[-_]", raw)
        if prefix and prefix.group(1) in self.by_paper:
            return prefix.group(1)
        examples = ", ".join(sorted(self.by_paper, key=lambda key: (len(key), key))[:3])
        raise UnknownIdentifier(
            f"unknown paper_id {raw!r}. Use a local paper key such as {examples} (list_papers shows all), "
            "a verified DBLP record IRI (https://dblp.org/rec/...), or a source id like 'p23-metadata'."
        )

    def paper_keys(self, values: list[str] | None) -> list[str] | None:
        return None if values is None else [self.paper_key(value) for value in values]

    def normalise_scope(self, scope: dict | None) -> dict | None:
        if not scope or not scope.get("paper_ids"):
            return scope
        return {**scope, "paper_ids": self.paper_keys(list(scope["paper_ids"]))}

    def fact_predicates(self) -> list[str]:
        return sorted({str(row.get("predicate")) for row in self.fact_rows if row.get("predicate")})

    def check_predicate(self, predicate: str | None) -> str | None:
        if predicate is None:
            return None
        aliases = {"authoredby": "author", "authored_by": "author", "authors": "author",
                   "yearofpublication": "year", "publishedin": "venue"}
        normalised = aliases.get(predicate.casefold().replace(" ", ""), predicate)
        known = self.fact_predicates()
        if normalised not in known:
            raise UnknownIdentifier(f"unknown predicate {predicate!r}. Available predicates: {', '.join(known)}.")
        return normalised

    def distinct_works(self, rows: list[dict]) -> list[dict]:
        """A preprint and its near-identical journal version count as one work."""
        works: list[tuple[str, dict]] = []
        for row in sorted(rows, key=lambda item: str(item["subject"])):
            title = self.paper_title(str(row["subject"])) or str(row["subject"])
            normalized = " ".join(re.findall(r"[a-z0-9]+", title.casefold()))
            if any(SequenceMatcher(None, normalized, prior).ratio() >= 0.98 for prior, _ in works):
                continue
            works.append((normalized, row))
        return [row for _, row in works]

    # ─── Corpus-wide metadata ────────────────────────────────────────────────

    def metadata_table(
        self, predicates: list[str] | None = None, paper_ids: list[str] | None = None,
        offset: int = 0, limit: int = 500,
    ) -> dict:
        """Every fact of the given predicates for every (or the given) paper, in one call."""
        wanted = {self.check_predicate(item) for item in predicates} if predicates else None
        keys = set(self.paper_keys(paper_ids) or []) or None
        rows = [
            {
                "paper_id": str(row["subject"]),
                "dblp_iri": self.canonical_paper_iri(str(row["subject"])),
                "title": self.paper_title(str(row["subject"])),
                "predicate": row["predicate"], "object": row["object"],
                "fact_id": row.get("fact_id"), "source_id": row["source_id"],
            }
            for row in self.fact_rows
            if (wanted is None or row.get("predicate") in wanted)
            and (keys is None or str(row["subject"]) in keys)
            and str(row["subject"]) in self.by_paper
        ]
        rows.sort(key=lambda row: (row["paper_id"], row["predicate"], str(row["object"])))
        return paged(rows, offset, limit)

    def aggregate(
        self, operation: str, predicate: str | None = None, top_k: int | None = None,
        min_count: int | None = None,
    ) -> dict:
        """Deterministic corpus-wide aggregates, each with the fact/reference ids it used.

        operation:
          count_papers_by_author   distinct works per author (preprint + journal version = one)
          count_by_value           papers per value of ``predicate`` (e.g. year, venue)
          min / max                smallest / largest numeric value of ``predicate`` (e.g. year)
          count_incoming_citations in-corpus citations received per paper
          count_citations_by_author in-corpus citations received by each author's distinct works
        """
        facts = [row for row in self.fact_rows if str(row.get("subject")) in self.by_paper]
        items: list[dict] = []
        if operation == "count_papers_by_author":
            grouped: dict[str, list[dict]] = {}
            for row in facts:
                if row.get("predicate") == "author":
                    grouped.setdefault(str(row["object"]), []).append(row)
            for author, rows in grouped.items():
                works = self.distinct_works(rows)
                items.append({"author": author, "count": len(works),
                              "paper_ids": sorted(str(row["subject"]) for row in works),
                              "fact_ids": sorted(str(row.get("fact_id") or row["source_id"]) for row in works),
                              "source_ids": sorted({str(row["source_id"]) for row in works})})
            unit = "distinct papers"
        elif operation == "count_by_value":
            key = self.check_predicate(predicate or "")
            grouped = {}
            for row in facts:
                if row.get("predicate") == key:
                    grouped.setdefault(str(row["object"]), []).append(row)
            items = [{"value": value, "count": len({str(row["subject"]) for row in rows}),
                      "paper_ids": sorted({str(row["subject"]) for row in rows}),
                      "fact_ids": sorted(str(row.get("fact_id") or row["source_id"]) for row in rows),
                      "source_ids": sorted({str(row["source_id"]) for row in rows})}
                     for value, rows in grouped.items()]
            unit = "papers"
        elif operation in {"min", "max"}:
            key = self.check_predicate(predicate or "")
            numeric = [row for row in facts if row.get("predicate") == key
                       and re.fullmatch(r"-?\d+(?:\.\d+)?", str(row["object"]))]
            if not numeric:
                raise UnknownIdentifier(f"predicate {predicate!r} has no numeric values; try 'year'.")
            pick = min if operation == "min" else max
            best = pick(float(row["object"]) for row in numeric)
            rows = [row for row in numeric if float(row["object"]) == best]
            items = [{"value": rows[0]["object"], "count": len(rows),
                      "paper_ids": sorted(str(row["subject"]) for row in rows),
                      "fact_ids": sorted(str(row.get("fact_id") or row["source_id"]) for row in rows),
                      "source_ids": sorted({str(row["source_id"]) for row in rows})}]
            unit = f"{operation} {key}"
        elif operation in {"count_incoming_citations", "count_citations_by_author"}:
            incoming: dict[str, list[dict]] = {}
            for reference in self.references:
                target = reference.get("resolved_paper_id")
                if target and target in self.by_paper:
                    incoming.setdefault(str(target), []).append(reference)
            if operation == "count_incoming_citations":
                items = [{"paper_id": paper, "title": self.paper_title(paper),
                          "count": len({str(ref["paper_id"]) for ref in refs}),
                          "citing_paper_ids": sorted({str(ref["paper_id"]) for ref in refs}),
                          "reference_ids": sorted(str(ref["reference_id"]) for ref in refs)}
                         for paper, refs in incoming.items()]
                unit = "citing papers"
            else:
                grouped = {}
                for row in facts:
                    if row.get("predicate") == "author":
                        grouped.setdefault(str(row["object"]), []).append(row)
                for author, rows in grouped.items():
                    works = self.distinct_works(rows)
                    refs = [ref for row in works for ref in incoming.get(str(row["subject"]), [])]
                    items.append({"author": author, "count": len({str(ref["paper_id"]) for ref in refs}),
                                  "paper_ids": sorted(str(row["subject"]) for row in works),
                                  "reference_ids": sorted(str(ref["reference_id"]) for ref in refs),
                                  "fact_ids": sorted(str(row.get("fact_id") or row["source_id"]) for row in works)})
                unit = "citing papers (in-corpus) across the author's distinct papers"
        else:
            raise UnknownIdentifier(
                f"unknown operation {operation!r}. Use count_papers_by_author, count_by_value, min, max, "
                "count_incoming_citations or count_citations_by_author."
            )
        items.sort(key=lambda item: (-int(item["count"]), str(item.get("author") or item.get("value") or item.get("paper_id"))))
        if min_count is not None:
            items = [item for item in items if item["count"] >= min_count]
        if top_k is not None and operation not in {"min", "max"} and items and top_k >= 1:
            cutoff = items[min(top_k, len(items)) - 1]["count"]
            items = [item for item in items if item["count"] >= cutoff]   # keep ties
        return {"operation": operation, "unit": unit, "computed": True, "total": len(items), "items": items,
                "coverage": {"scope": "all_indexed_papers", "exhaustive": True,
                             "papers_total": len(self.by_paper), "papers_examined": len(self.by_paper)}}

    def canonical_paper_iri(self, value: str) -> str | None:
        """Return a verified DBLP IRI for an internal paper key or IRI."""
        raw = str(value).strip()
        if raw.startswith("https://dblp.org/rec/"):
            return raw if raw in self.by_dblp else None
        paper = self.by_paper.get(raw)
        iri = paper.get("dblp_iri") if paper else None
        return str(iri) if iri else None

    def paper_title(self, value: str) -> str | None:
        raw = str(value).strip()
        paper = self.by_paper.get(raw)
        if paper:
            return str(paper.get("title") or "") or None
        pid = self.by_dblp.get(raw)
        return str(self.by_paper[pid].get("title") or "") if pid else None

    def mentioned_paper_ids(self, question: str) -> list[str]:
        """Resolve explicit title prefixes and distinctive method names in a question."""
        words = re.findall(r"[a-z0-9]+", question.casefold())
        normalized_question = " ".join(words)
        matched = []
        for paper_id, paper in self.by_paper.items():
            title = str(paper.get("title") or "")
            title_words = re.findall(r"[a-z0-9]+", title.casefold())
            prefix = " ".join(title_words[:3])
            brand = re.split(r"[^A-Za-z0-9]+", title, maxsplit=1)[0]
            distinctive = (
                len(brand) >= 4
                and (brand.isupper() or any(char.isdigit() for char in brand) or any(char.isupper() for char in brand[1:]))
            )
            if (
                len(title_words) >= 3 and prefix in normalized_question
                or distinctive and re.search(rf"(?i)\b{re.escape(brand)}\b", question)
            ):
                matched.append(paper_id)
        return matched

    def rank_evidence(
        self,
        query: str,
        passage_ids: list[str] | None = None,
        paper_ids: list[str] | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> dict:
        """Rank existing passages with the corpus BM25S index, retaining scores."""
        allowed = set(passage_ids) if passage_ids is not None else set(self.by_passage)
        papers = set(paper_ids) if paper_ids is not None else None
        allowed = {
            passage_id for passage_id in allowed
            if passage_id in self.by_passage
            and (papers is None or self.by_passage[passage_id]["paper_id"] in papers)
        }
        if self.bm25 is not None and allowed:
            tokens = bm25s.tokenize([query], stopwords="en", show_progress=False)
            ranked = self.bm25.retrieve(
                tokens, k=len(self.passages), show_progress=False, sorted=True
            )
            rows = [
                {**self.by_passage[doc["passage_id"]], "score": float(score)}
                for doc, score in zip(ranked.documents[0], ranked.scores[0], strict=True)
                if doc["passage_id"] in allowed
            ]
        else:
            terms = [term for term in re.findall(r"[a-z0-9]+", query.casefold()) if len(term) > 2]
            rows = [
                {**self.by_passage[passage_id], "score": float(sum(self.by_passage[passage_id]["text"].casefold().count(term) for term in terms))}
                for passage_id in allowed
            ]
            rows.sort(key=lambda row: (-row["score"], row["passage_id"]))
        return paged(rows, offset, limit)

    # ─── Ranked retrieval (BM25 + dense) ─────────────────────────────────────

    def _bm25_order(self, query: str, allowed: set[str]) -> list[tuple[str, float]]:
        """Allowed passages with a positive BM25 score, best first (no boolean filter)."""
        if not allowed:
            return []
        if self.bm25 is not None:
            tokens = bm25s.tokenize([query], stopwords="en", show_progress=False)
            if not tokens.vocab:
                return []
            ranked = self.bm25.retrieve(tokens, k=len(self.passages), show_progress=False, sorted=True)
            return [(doc["passage_id"], float(score))
                    for doc, score in zip(ranked.documents[0], ranked.scores[0], strict=True)
                    if score > 0 and doc["passage_id"] in allowed]
        # No BM25 index (small test corpora): count content words of four letters or more.
        terms = [term for term in re.findall(r"[a-z0-9]+", query.casefold()) if len(term) > 3]
        scored = [(pid, float(sum(self.by_passage[pid]["text"].casefold().count(t) for t in terms)))
                  for pid in allowed]
        return sorted((row for row in scored if row[1] > 0), key=lambda row: (-row[1], row[0]))

    def embed_query(self, query: str) -> np.ndarray | None:
        """The query's embedding, or None when dense retrieval is off or unavailable.

        Dense retrieval is on only with ``HARNESS_RETRIEVAL_MODE=hybrid`` and a current index, so one
        setting switches every ranked search (pipeline, concept evidence, ``search_evidence``, paper
        readers) between BM25 alone and BM25 + dense.
        """
        if settings().retrieval_mode != "hybrid" or not self.dense.available:
            return None
        return _cached_query_embedding(query, str(self.dense.meta.get("model") or settings().model_embedding))

    def _rerank(self, query: str, ranked: list[str]) -> list[str] | None:
        """Reorder the top fused candidates with a served reranker (LiteLLM ``/rerank``), if configured."""
        model = settings().model_rerank.strip()
        if not model or not ranked:
            return None
        head = ranked[:RERANK_DEPTH]
        try:
            response = httpx.post(
                f"{settings().litellm_base_url.rstrip('/')}/rerank",
                headers={"Authorization": f"Bearer {settings().api_key}"},
                json={"model": model, "query": query, "top_n": len(head),
                      "documents": [self.by_passage[pid]["text"][:4000] for pid in head]},
                timeout=30,
            )
            response.raise_for_status()
            results = response.json()["results"]
            if all("relevance_score" in item for item in results):
                results = sorted(results, key=lambda item: -float(item["relevance_score"]))
            order = list(dict.fromkeys(head[int(item["index"])] for item in results))
            if not order:
                raise ValueError("empty rerank result")
        except (httpx.HTTPError, KeyError, ValueError, IndexError, TypeError) as exc:
            LOGGER.warning("Reranker %s unavailable (%s); keeping the fused order", model, exc)
            return None
        placed = set(order)
        return order + [pid for pid in ranked if pid not in placed]

    def hybrid_rank(
        self, query: str, scope: dict | None = None, passage_ids: list[str] | None = None,
        offset: int = 0, limit: int = 10,
    ) -> dict:
        """Ranked retrieval for a natural-language query: BM25 ∪ dense, fused, optionally reranked.

        Items are passages with ``score`` (fused), ``bm25_rank`` and ``dense_rank`` (None when that
        ranking did not return the passage). ``retrieval`` says which rankings took part.
        """
        restrict = set(passage_ids) if passage_ids is not None else None
        allowed = {
            p["passage_id"] for p in self.passages
            if in_scope(self.by_paper[p["paper_id"]], scope)
            and (restrict is None or p["passage_id"] in restrict)
            and (not (scope or {}).get("section_roles")
                 or any(role.casefold() in p["section_path"].casefold() for role in scope["section_roles"]))
        }
        bm25 = [pid for pid, _ in self._bm25_order(query, allowed)[:FUSION_DEPTH]]
        vector = self.embed_query(query) if allowed else None
        dense: list[str] = []
        if vector is not None:
            try:
                dense = [pid for pid, _ in self.dense.search(vector, allowed, FUSION_DEPTH)]
            except (ValueError, OSError) as exc:  # e.g. the embedding model changed dimension
                LOGGER.warning("Dense search failed (%s); using BM25 only", exc)
        fused = rrf([ranking for ranking in (bm25, dense) if ranking])
        order = [pid for pid, _ in fused]
        reranked = self._rerank(query, order)
        method = "+".join(name for name, ranking in (("bm25", bm25), ("dense", dense)) if ranking) or "none"
        if reranked is not None:
            order, method = reranked, method + "+rerank"
        scores = dict(fused)
        bm25_rank = {pid: rank for rank, pid in enumerate(bm25, start=1)}
        dense_rank = {pid: rank for rank, pid in enumerate(dense, start=1)}
        rows = [{**self.by_passage[pid], "score": round(scores.get(pid, 0.0), 6),
                 "bm25_rank": bm25_rank.get(pid), "dense_rank": dense_rank.get(pid)} for pid in order]
        return {**paged(rows, offset, limit), "retrieval": method, "passages_in_scope": len(allowed)}

    def reference_title(self, reference_id: str) -> str | None:
        row = self.by_reference.get(str(reference_id))
        if not row:
            return None
        return str(row.get("title") or row.get("raw") or "") or None

    def list_papers(self, scope: dict | None = None, offset: int = 0, limit: int = 10) -> dict:
        return paged([p for p in self.papers if in_scope(p, scope)], offset, limit)

    def abstracts(self, paper_ids: list[str], offset: int = 0, limit: int = 10) -> dict:
        return paged(
            [
                {
                    "paper_id": pid,
                    "abstract": self.by_paper[pid]["abstract"],
                    "passage_id": f"{pid}-abstract"
                    if f"{pid}-abstract" in self.by_passage
                    else None,
                }
                for pid in paper_ids
                if pid in self.by_paper
            ],
            offset,
            limit,
        )

    def paper_outline(self, pid: str) -> dict:
        path = self.root / "outlines" / f"{pid}.md"
        if not path.exists():
            raise KeyError(pid)
        return {
            "total": 1,
            "offset": 0,
            "limit": 1,
            "items": [
                {"paper_id": pid, "outline": path.read_text(), "source_id": f"{pid}-outline"}
            ],
        }

    def search(self, dsl: str, scope: dict | None = None, offset: int = 0, limit: int = 10) -> dict:
        scope = dict(scope or {})
        fields = re.findall(r"\b(paper|section|year):([^\s()]+)", dsl, re.IGNORECASE)
        for key, value in fields:
            if key.lower() == "paper":
                scope["paper_ids"] = [value]
            elif key.lower() == "year":
                scope["year_range"] = [int(value), int(value)]
            elif key.lower() == "section":
                scope["section_roles"] = [value]
        query = re.sub(r"\b(?:paper|section|year):[^\s()]+", "", dsl, flags=re.IGNORECASE).strip()
        if not query:
            raise ValueError("search requires content terms")
        candidates = [
            p
            for p in self.passages
            if in_scope(self.by_paper[p["paper_id"]], scope)
            and (
                not scope.get("section_roles")
                or any(
                    role.casefold() in p["section_path"].casefold()
                    for role in scope["section_roles"]
                )
            )
        ]
        found = {p["passage_id"]: p for p in candidates if matches(query, p["text"])}
        if self.bm25 is not None and found:
            terms = " ".join(
                t
                for t in lex(query)
                if t.upper() not in {"AND", "OR", "NOT"} and not t.upper().startswith("NEAR/")
            )
            tokens = bm25s.tokenize([terms], stopwords="en", show_progress=False)
            ranked = self.bm25.retrieve(
                tokens, k=len(self.passages), show_progress=False, sorted=True
            ).documents[0]
            rows = [found[item["passage_id"]] for item in ranked if item["passage_id"] in found]
        else:
            rows = list(found.values())
        return paged(rows, offset, limit)

    def grep(self, regex: str, scope: dict | None = None, offset: int = 0, limit: int = 10) -> dict:
        pattern = BoundedPattern(regex)
        rows = []
        for passage in self.passages:
            if not in_scope(self.by_paper[passage["paper_id"]], scope):
                continue
            if (
                scope
                and scope.get("section_roles")
                and not any(
                    role.casefold() in passage["section_path"].casefold()
                    for role in scope["section_roles"]
                )
            ):
                continue
            match = pattern.search(passage["text"])
            if match:
                rows.append(
                    {
                        "passage_id": passage["passage_id"],
                        "paper_id": passage["paper_id"],
                        "line": passage["text"],
                        "match_start": match.start(),
                        "match_end": match.end(),
                    }
                )
        return paged(rows, offset, limit)

    def read_passage(self, passage_id: str, window: int = 0) -> dict:
        passage = self.by_passage[passage_id]
        siblings = [p for p in self.passages if p["paper_id"] == passage["paper_id"]]
        idx = next(i for i, p in enumerate(siblings) if p["passage_id"] == passage_id)
        items = siblings[max(0, idx - window) : idx + window + 1]
        return {"total": len(items), "offset": 0, "limit": len(items), "items": items}

    def facts(
        self, paper_id: str, offset: int = 0, limit: int = 10,
        predicate: str | None = None,
    ) -> dict:
        paper_id = self.paper_key(paper_id)
        predicate = self.check_predicate(predicate)
        return paged(
            [
                fact for fact in self.fact_rows
                if fact["subject"] == paper_id
                and (predicate is None or fact["predicate"] == predicate)
            ],
            offset, limit,
        )

    def citations(
        self, paper_id: str, direction: str = "out", offset: int = 0, limit: int = 10,
        in_corpus_only: bool = False,
    ) -> dict:
        if direction not in {"in", "out"}:
            raise ValueError("direction must be in or out")
        paper_id = self.paper_key(paper_id)
        references = [
            r
            for r in self.references
            if (r["paper_id"] if direction == "out" else r["resolved_paper_id"]) == paper_id
            and (not in_corpus_only or bool(r.get("resolved_paper_id")))
        ]
        rows = [
            {**reference, "fact_id": f"{reference['reference_id']}-cites"}
            for reference in references
        ]
        return paged(rows, offset, limit)

    def bulk_check(
        self, pattern: str, paper_ids: list[str], offset: int = 0, limit: int = 10
    ) -> dict:
        regex = BoundedPattern(pattern)
        rows = []
        for pid in paper_ids:
            hits = [p for p in self.passages if p["paper_id"] == pid and regex.search(p["text"])]
            rows.append(
                {
                    "paper_id": pid,
                    "hit": bool(hits),
                    "passage_id": hits[0]["passage_id"] if hits else None,
                    "line": hits[0]["text"] if hits else None,
                }
            )
        return paged(rows, offset, limit)

    def keyphrases(self, scope: dict | None = None, offset: int = 0, limit: int = 10) -> dict:
        """Return corpus-local lexical keyphrases with supporting passage IDs."""
        rows = [
            passage for passage in self.passages
            if in_scope(self.by_paper[passage["paper_id"]], scope)
        ]
        counts: Counter[str] = Counter()
        support: dict[str, set[str]] = {}
        for passage in rows:
            words = re.findall(r"[A-Za-z][A-Za-z0-9-]{3,}", passage["text"].casefold())
            seen: set[str] = set()
            for size in (2, 3):
                for index in range(len(words) - size + 1):
                    phrase = " ".join(words[index : index + size])
                    if phrase in seen or any(token in {"this", "that", "with", "from", "which"} for token in phrase.split()):
                        continue
                    seen.add(phrase)
                    counts[phrase] += 1
                    support.setdefault(phrase, set()).add(passage["passage_id"])
        ranked = [
            {"phrase": phrase, "count": count, "passage_ids": sorted(support[phrase])[:10]}
            for phrase, count in counts.most_common()
            if count > 1
        ]
        return paged(ranked, offset, limit)

    def results(self, pattern: str, scope: dict | None = None, offset: int = 0, limit: int = 10) -> dict:
        """Find numeric result statements while retaining their evidence IDs."""
        regex = BoundedPattern(pattern)
        rows = []
        for passage in self.passages:
            if not in_scope(self.by_paper[passage["paper_id"]], scope):
                continue
            for sentence in re.split(r"(?<=[.!?])\s+", passage["text"]):
                if regex.search(sentence) and re.search(r"\d", sentence):
                    rows.append({"passage_id": passage["passage_id"], "paper_id": passage["paper_id"], "text": sentence.strip()})
        return paged(rows, offset, limit)

    def expand(self, passage_id: str, window: int = 2) -> dict:
        """Expand a passage to bounded neighbouring evidence."""
        if window < 0 or window > 5:
            raise ValueError("window must be between 0 and 5")
        return self.read_passage(passage_id, window)

    def glossary(self, term: str | None = None, offset: int = 0, limit: int = 10) -> dict:
        """Return corpus terms with stable supporting fact or passage IDs."""
        rows = []
        for value in sorted({str(f.get("object")) for f in self.fact_rows if f.get("object")}):
            if term and term.casefold() not in value.casefold():
                continue
            support = [str(f.get("source_id")) for f in self.fact_rows if str(f.get("object")) == value]
            rows.append({"term": value, "source_ids": sorted(set(support))})
        return paged(rows, offset, limit)

    def table(self, paper_id: str, table_id: str | None = None, offset: int = 0, limit: int = 10) -> dict:
        path = self.root / "tables" / f"{paper_id}.json"
        rows = read_jsonl(path) if path.suffix == ".jsonl" else (json.loads(path.read_text()) if path.exists() else [])
        if isinstance(rows, dict):
            rows = rows.get("tables", [])
        if table_id:
            rows = [row for row in rows if str(row.get("table_id", row.get("id"))) == table_id]
        return paged(rows, offset, limit)

    def table_search(self, query: str, scope: dict | None = None, offset: int = 0, limit: int = 10) -> dict:
        rows = []
        for paper in self.papers:
            if not in_scope(paper, scope):
                continue
            table_rows = self.table(paper["paper_id"], limit=50)["items"]
            for row in table_rows:
                if query.casefold() in json.dumps(row).casefold():
                    rows.append({"paper_id": paper["paper_id"], **row})
        return paged(rows, offset, limit)

    def section(self, role: str, scope: dict | None = None, offset: int = 0, limit: int = 10) -> dict:
        rows = [p for p in self.passages if in_scope(self.by_paper[p["paper_id"]], scope) and role.casefold() in p["section_path"].casefold()]
        return paged(rows, offset, limit)

    def citation_contexts(self, paper_id: str, offset: int = 0, limit: int = 10) -> dict:
        rows = []
        for reference in self.references:
            if reference.get("paper_id") != paper_id:
                continue
            rows.append({**reference, "context": self.by_passage.get(reference.get("passage_id", ""), {}).get("text", "")})
        return paged(rows, offset, limit)

    def query2doc(self, query: str, offset: int = 0, limit: int = 10) -> dict:
        """Natural-language query: the same ranked retrieval as ``hybrid_rank``."""
        return self.hybrid_rank(query, offset=offset, limit=limit)

    def find_number(self, pattern: str, scope: dict | None = None, offset: int = 0, limit: int = 10) -> dict:
        return self.results(pattern, scope, offset, limit)

    def co_cited(self, paper_id: str, offset: int = 0, limit: int = 10) -> dict:
        cited = {r.get("resolved_paper_id") for r in self.references if r.get("paper_id") == paper_id}
        rows = [{"paper_id": pid, "co_cited_with": sorted(cited - {pid})} for pid in sorted(cited) if pid]
        return paged(rows, offset, limit)

    def rank(self, query: str, scope: dict | None = None, offset: int = 0, limit: int = 10) -> dict:
        """Ranked passages for a query (kept for compatibility; same as ``hybrid_rank``)."""
        result = self.hybrid_rank(query, scope, offset=offset, limit=limit)
        return {**result, "items": [{"passage_id": row["passage_id"], "paper_id": row["paper_id"],
                                     "score": row["score"]} for row in result["items"]]}

    def entities(self, scope: dict | None = None, offset: int = 0, limit: int = 10) -> dict:
        counts: Counter[str] = Counter()
        support: dict[str, set[str]] = {}
        for passage in self.passages:
            if not in_scope(self.by_paper[passage["paper_id"]], scope):
                continue
            for entity in re.findall(r"\b[A-Z][A-Za-z0-9-]{2,}(?:\s+[A-Z][A-Za-z0-9-]{2,})*", passage["text"]):
                counts[entity] += 1
                support.setdefault(entity, set()).add(passage["passage_id"])
        rows = [{"entity": entity, "count": count, "passage_ids": sorted(support[entity])[:10]} for entity, count in counts.most_common()]
        return paged(rows, offset, limit)

    def entity_index(self, entity: str, scope: dict | None = None, offset: int = 0, limit: int = 10) -> dict:
        return self.grep(rf"\b{re.escape(entity)}\b", scope, offset, limit)

    def candidates_for(self, question: str, offset: int = 0, limit: int = 10) -> dict:
        return self.rank(question, offset=offset, limit=limit)

    def prior_triples(self, paper_ids: list[str] | None = None, offset: int = 0, limit: int = 10) -> dict:
        rows = [row for row in self.seed_facts() if not paper_ids or str(row.get("subject")) in paper_ids]
        return paged(rows, offset, limit)

    def corpus_map(self, scope: dict | None = None, offset: int = 0, limit: int = 10) -> dict:
        rows = []
        for paper in self.papers:
            if in_scope(paper, scope):
                rows.append({"paper_id": paper["paper_id"], "title": paper.get("title", ""), "year": paper.get("year"), "passages": sum(p["paper_id"] == paper["paper_id"] for p in self.passages)})
        return paged(rows, offset, limit)


EMBED_RETRY_S = 60.0   # after a failed query embedding, BM25 only for this long (no 20 s wait per query)
_embedding_down_until: dict[tuple[str, str], float] = {}


@lru_cache(maxsize=512)
def _embed_query_cached(query: str, model: str, base_url: str) -> np.ndarray:
    """Successful embeddings only: a failure raises, and lru_cache does not keep exceptions."""
    del base_url  # part of the cache key only
    return embed_texts([QUERY_INSTRUCTION + query], model=model, timeout_s=20)[0]


def _cached_query_embedding(query: str, model: str) -> np.ndarray | None:
    base_url = settings().litellm_base_url
    if time.monotonic() < _embedding_down_until.get((model, base_url), 0.0):
        return None
    try:
        return _embed_query_cached(query, model, base_url)
    except (httpx.HTTPError, KeyError, ValueError, IndexError, TypeError) as exc:
        _embedding_down_until[(model, base_url)] = time.monotonic() + EMBED_RETRY_S
        LOGGER.warning("Query embedding with %s failed (%s); BM25 only for %.0f s", model, exc, EMBED_RETRY_S)
        return None
