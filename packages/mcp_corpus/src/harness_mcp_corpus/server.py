"""M1 corpus tools over MCP; local stdio or Streamable HTTP."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from harness_corpus.store import CorpusStore
from mcp.server.fastmcp import FastMCP


def create_server(root: Path, host: str = "127.0.0.1", port: int = 8765) -> FastMCP:
    corpus = CorpusStore(root)
    mcp = FastMCP("corpus", host=host, port=port)

    def coverage(
        result: dict[str, Any], *, scope: dict | None = None, exhaustive: bool = False,
        passages_examined: int | None = None,
    ) -> dict[str, Any]:
        """Attach coverage without changing the existing corpus search semantics."""
        scoped_papers = [paper for paper in corpus.papers if not scope or not scope.get("paper_ids") or paper["paper_id"] in scope["paper_ids"]]
        return {
            **result,
            "coverage": {
                "scope": "selected_indexed_papers" if scope and scope.get("paper_ids") else "all_indexed_papers",
                "exhaustive": exhaustive,
                "papers_total": len(scoped_papers),
                "papers_examined": len(scoped_papers) if exhaustive else None,
                "passages_examined": passages_examined if passages_examined is not None else len(result.get("items", [])),
                "candidate_count": int(result.get("total", len(result.get("items", [])))),
            },
        }

    def claim_terms(claim: dict[str, Any]) -> tuple[str, dict[str, list[str]]]:
        fields = claim.get("fields", claim)
        if not isinstance(fields, dict):
            raise ValueError("claim must be an object or contain a fields object")  # noqa: TRY004
        terms_by_field: dict[str, list[str]] = {}
        for name, value in fields.items():
            if isinstance(value, dict):
                value = value.get("value", "")
            if value is None:
                continue
            terms = [term for term in re.findall(r"[A-Za-z][A-Za-z0-9_.-]*|\d+(?:\.\d+)?%?", str(value)) if len(term) > 1]
            if terms:
                terms_by_field[str(name)] = terms
        query = " ".join(term for terms in terms_by_field.values() for term in terms)
        if not query:
            raise ValueError("claim contains no searchable fields")
        return query, terms_by_field

    @mcp.tool(description="List indexed papers with stable IDs and metadata.")
    def list_papers(scope: dict | None = None, offset: int = 0, limit: int = 10) -> dict:
        return corpus.list_papers(scope, offset, limit)

    @mcp.tool(description="Read a paper's title, abstract, headings, and section opening lines.")
    def paper_outline(paper_id: str) -> dict:
        return corpus.paper_outline(paper_id)

    @mcp.tool(description="Read abstracts by paper ID; each result carries a stable abstract ID.")
    def abstracts(paper_ids: list[str], offset: int = 0, limit: int = 10) -> dict:
        return corpus.abstracts(paper_ids, offset, limit)

    @mcp.tool(
        description="Find passages with lexical boolean, phrase, proximity, or field query terms."
    )
    def search(dsl: str, scope: dict | None = None, offset: int = 0, limit: int = 10) -> dict:
        return corpus.search(dsl, scope, offset, limit)

    @mcp.tool(description="Find regex matches with passage IDs and matching lines.")
    def grep(regex: str, scope: dict | None = None, offset: int = 0, limit: int = 10) -> dict:
        return corpus.grep(regex, scope, offset, limit)

    @mcp.tool(description="Read a passage with neighbouring passages from the same paper.")
    def read_passage(passage_id: str, window: int = 0) -> dict:
        return corpus.read_passage(passage_id, window)

    @mcp.tool(description="Read structural facts for one paper with source fact IDs. Filter by predicate, for example author, title, year, or venue, to avoid unrelated facts.")
    def facts(
        paper_id: str, offset: int = 0, limit: int = 10,
        predicate: str | None = None,
    ) -> dict:
        return corpus.facts(paper_id, offset, limit, predicate)

    @mcp.tool(
        description="Read incoming or outgoing citations with citation fact IDs. Use indexed_only=true for corpus-scoped questions. An exhaustive indexed incoming zero may support corpus absence."
    )
    def citations(
        paper_id: str, direction: str = "out", offset: int = 0, limit: int = 10,
        in_corpus_only: bool = False, indexed_only: bool | None = None,
    ) -> dict:
        indexed = in_corpus_only if indexed_only is None else indexed_only
        result = corpus.citations(paper_id, direction, offset, limit, indexed)
        exhaustive = bool(indexed and direction == "in" and offset == 0 and limit >= result.get("total", 0))
        return coverage(result, exhaustive=exhaustive)

    @mcp.tool(description="Check a regex across papers, returning one hit or no-hit per paper.")
    def bulk_check(pattern: str, paper_ids: list[str], offset: int = 0, limit: int = 10) -> dict:
        return corpus.bulk_check(pattern, paper_ids, offset, limit)

    @mcp.tool(description="Rank repeated corpus-local keyphrases with supporting passage IDs.")
    def keyphrases(scope: dict | None = None, offset: int = 0, limit: int = 10) -> dict:
        return corpus.keyphrases(scope, offset, limit)

    @mcp.tool(description="Find numeric result statements matching a pattern with passage IDs.")
    def results(pattern: str, scope: dict | None = None, offset: int = 0, limit: int = 10) -> dict:
        return corpus.results(pattern, scope, offset, limit)

    @mcp.tool(description="Expand one passage to bounded neighbouring passages.")
    def expand(passage_id: str, window: int = 2) -> dict:
        return corpus.expand(passage_id, window)

    @mcp.tool(description="List grounded corpus glossary terms.")
    def glossary(term: str | None = None, offset: int = 0, limit: int = 10) -> dict:
        return corpus.glossary(term, offset, limit)

    @mcp.tool(description="Read an indexed table for a paper.")
    def table(paper_id: str, table_id: str | None = None, offset: int = 0, limit: int = 10) -> dict:
        return corpus.table(paper_id, table_id, offset, limit)

    @mcp.tool(description="Search indexed table cells or captions.")
    def table_search(query: str, scope: dict | None = None, offset: int = 0, limit: int = 10) -> dict:
        return corpus.table_search(query, scope, offset, limit)

    @mcp.tool(description="Read passages belonging to a section role.")
    def section(role: str, scope: dict | None = None, offset: int = 0, limit: int = 10) -> dict:
        return corpus.section(role, scope, offset, limit)

    @mcp.tool(description="Read citation contexts with passage IDs.")
    def citation_contexts(paper_id: str, offset: int = 0, limit: int = 10) -> dict:
        return corpus.citation_contexts(paper_id, offset, limit)

    @mcp.tool(description="Search the corpus from a natural-language query.")
    def query2doc(query: str, offset: int = 0, limit: int = 10) -> dict:
        return corpus.query2doc(query, offset, limit)

    @mcp.tool(description="Find numeric evidence matching a pattern.")
    def find_number(pattern: str, scope: dict | None = None, offset: int = 0, limit: int = 10) -> dict:
        return corpus.find_number(pattern, scope, offset, limit)

    @mcp.tool(description="Find papers sharing cited references.")
    def co_cited(paper_id: str, offset: int = 0, limit: int = 10) -> dict:
        return corpus.co_cited(paper_id, offset, limit)

    @mcp.tool(description="Rank passages lexically for a query.")
    def rank(query: str, scope: dict | None = None, offset: int = 0, limit: int = 10) -> dict:
        return corpus.rank(query, scope, offset, limit)

    @mcp.tool(description="Rank retrieved evidence with BM25S scores, optionally within selected papers or passage IDs.")
    def rank_evidence(
        query: str, passage_ids: list[str] | None = None,
        paper_ids: list[str] | None = None, offset: int = 0, limit: int = 50,
    ) -> dict:
        return corpus.rank_evidence(query, passage_ids, paper_ids, offset, limit)

    @mcp.tool(description="List named entities with supporting passages.")
    def entities(scope: dict | None = None, offset: int = 0, limit: int = 10) -> dict:
        return corpus.entities(scope, offset, limit)

    @mcp.tool(description="Find passages mentioning one entity.")
    def entity_index(entity: str, scope: dict | None = None, offset: int = 0, limit: int = 10) -> dict:
        return corpus.entity_index(entity, scope, offset, limit)

    @mcp.tool(description="Rank candidate evidence for a question.")
    def candidates_for(question: str, offset: int = 0, limit: int = 10) -> dict:
        return corpus.candidates_for(question, offset, limit)

    @mcp.tool(description="Read prior seeded triples for selected papers.")
    def prior_triples(paper_ids: list[str] | None = None, offset: int = 0, limit: int = 10) -> dict:
        return corpus.prior_triples(paper_ids, offset, limit)

    @mcp.tool(description="Map corpus papers to passage counts and metadata.")
    def corpus_map(scope: dict | None = None, offset: int = 0, limit: int = 10) -> dict:
        return corpus.corpus_map(scope, offset, limit)

    @mcp.tool(description="High-level agent evidence search. Composes existing lexical search and BM25 evidence ranking, returning stable passage IDs and coverage.")
    def search_evidence(query: str, scope: dict | None = None, limit: int = 20) -> dict:
        lexical = corpus.search(query, scope=scope, limit=max(1, len(corpus.passages)))
        ids = [str(item["passage_id"]) for item in lexical["items"]]
        ranked = corpus.rank_evidence(query, passage_ids=ids, limit=limit) if ids else {"items": [], "total": 0}
        return coverage(ranked, scope=scope, exhaustive=False, passages_examined=len(ids))

    @mcp.tool(description="High-level agent evidence reader. Read stable passage IDs with neighbouring context; raw evidence remains recoverable by ID.")
    def read_evidence(evidence_ids: list[str], window: int = 1) -> dict:
        items = [corpus.read_passage(evidence_id, window) for evidence_id in evidence_ids]
        return {"items": items, "total": len(items), "coverage": {"scope": "explicit_evidence_ids", "exhaustive": True, "papers_total": None, "papers_examined": None, "passages_examined": len(items), "candidate_count": len(items)}}

    @mcp.tool(description="High-level metadata operation. Return trusted structured facts with fact IDs and an explicit coverage envelope.")
    def structured_facts(paper_id: str, predicate: str | None = None, limit: int = 100) -> dict:
        return coverage(corpus.facts(paper_id, 0, limit, predicate), scope={"paper_ids": [paper_id]}, exhaustive=True)

    @mcp.tool(description="Gather candidate evidence for a proposed claim. It reports field-level support candidates and unresolved fields; it does not declare the claim true.")
    def evidence_for_claim(claim: dict[str, Any], scope: dict | None = None, limit: int = 20) -> dict:
        query, terms_by_field = claim_terms(claim)
        passages = search_evidence(query, scope, limit)
        tables = corpus.table_search(query, scope=scope, limit=limit)
        material = " ".join(
            json_text for item in passages["items"] + tables["items"]
            for json_text in [str(item).casefold()]
        )
        supported = [field for field, terms in terms_by_field.items() if any(term.casefold() in material for term in terms)]
        unresolved = [field for field in terms_by_field if field not in supported]
        return {
            "supported_fields": supported,
            "unresolved_fields": unresolved,
            "contradicted_fields": [],
            "supporting_passages": passages["items"],
            "supporting_tables": tables["items"],
            "coverage": {
                **passages["coverage"],
                "table_candidates": int(tables.get("total", 0)),
                "exhaustive": False,
            },
        }

    return mcp


def main() -> None:
    mcp = create_server(
        Path(os.getenv("CORPUS_ROOT", "./corpus_data")),
        os.getenv("MCP_CORPUS_HOST", "127.0.0.1"),
        int(os.getenv("MCP_CORPUS_PORT", "8765")),
    )
    transport = os.getenv("MCP_CORPUS_TRANSPORT", "streamable-http")
    if transport not in {"stdio", "streamable-http"}:
        raise ValueError("MCP_CORPUS_TRANSPORT must be stdio or streamable-http")
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
