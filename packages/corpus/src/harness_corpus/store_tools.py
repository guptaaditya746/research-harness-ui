"""The corpus tools the MCP server offers that read papers, passages, facts and references directly.

A mixin of ``CorpusStore`` (ranked retrieval and identifiers live in ``store.py``).
"""

from __future__ import annotations

import json
import re
from collections import Counter

import bm25s

from .dsl import BoundedPattern, lex, matches
from .rows import in_scope, paged, read_jsonl


class CorpusTools:
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
