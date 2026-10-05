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

import logging
import re
import time
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

import bm25s
import httpx
import numpy as np
from harness_config import settings

from .aggregates import aggregate
from .dense import QUERY_INSTRUCTION, DenseIndex, embed_texts, rrf
from .dsl import (  # noqa: F401 - re-exported
    PATTERN_BUDGET_S,
    PATTERN_MAX_LENGTH,
    BoundedPattern,
    lex,
    matches,
)
from .rows import UnknownIdentifier, in_scope, paged, read_jsonl
from .store_tools import CorpusTools

LOGGER = logging.getLogger(__name__)
FUSION_DEPTH = 200     # candidates taken from each ranking before fusion
RERANK_DEPTH = 30      # fused candidates sent to the optional reranker

class CorpusStore(CorpusTools):
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
        return aggregate(self, operation, predicate, top_k, min_count)

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
