"""Read-only corpus tools over ingest artifacts and a BM25S index."""

from __future__ import annotations

import json
import re
from pathlib import Path

import bm25s


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def paged(rows: list[dict], offset: int = 0, limit: int = 10) -> dict:
    if offset < 0 or limit < 1 or limit > 50:
        raise ValueError("offset must be nonnegative and limit must be 1..50")
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
        self.fact_rows = read_jsonl(root / "facts.jsonl")
        self.by_paper = {p["paper_id"]: p for p in self.papers}
        self.by_passage = {p["passage_id"]: p for p in self.passages}
        index_path = root / "index" / "bm25"
        self.bm25 = (
            bm25s.BM25.load(str(index_path), load_corpus=True)
            if (index_path / "params.index.json").exists()
            else None
        )

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

    def facts(self, paper_id: str, offset: int = 0, limit: int = 10) -> dict:
        return paged([f for f in self.fact_rows if f["subject"] == paper_id], offset, limit)

    def citations(
        self, paper_id: str, direction: str = "out", offset: int = 0, limit: int = 10
    ) -> dict:
        if direction not in {"in", "out"}:
            raise ValueError("direction must be in or out")
        references = [
            r
            for r in self.references
            if (r["paper_id"] if direction == "out" else r["resolved_paper_id"]) == paper_id
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
