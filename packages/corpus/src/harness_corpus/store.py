"""Read-only corpus tools over ingest artifacts and a BM25S index."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import bm25s


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


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

    def seed_facts(self) -> list[dict]:
        """Return deterministic, corpus-wide facts suitable for graph seeding."""
        return [
            row
            for row in self.fact_rows
            if row.get("predicate") != "cites" or row.get("in_corpus") is True
        ]

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
        pattern = re.compile(regex, re.IGNORECASE)
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
        regex = re.compile(pattern, re.IGNORECASE)
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
        regex = re.compile(pattern, re.IGNORECASE)
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
        return self.search(query, offset=offset, limit=limit)

    def find_number(self, pattern: str, scope: dict | None = None, offset: int = 0, limit: int = 10) -> dict:
        return self.results(pattern, scope, offset, limit)

    def co_cited(self, paper_id: str, offset: int = 0, limit: int = 10) -> dict:
        cited = {r.get("resolved_paper_id") for r in self.references if r.get("paper_id") == paper_id}
        rows = [{"paper_id": pid, "co_cited_with": sorted(cited - {pid})} for pid in sorted(cited) if pid]
        return paged(rows, offset, limit)

    def rank(self, query: str, scope: dict | None = None, offset: int = 0, limit: int = 10) -> dict:
        terms = [t.casefold() for t in re.findall(r"[A-Za-z0-9]+", query) if len(t) > 2]
        rows = []
        for passage in self.passages:
            if not in_scope(self.by_paper[passage["paper_id"]], scope):
                continue
            text = passage["text"].casefold()
            score = sum(text.count(term) for term in terms)
            if score:
                rows.append({"passage_id": passage["passage_id"], "paper_id": passage["paper_id"], "score": score})
        return paged(sorted(rows, key=lambda row: (-row["score"], row["passage_id"])), offset, limit)

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
