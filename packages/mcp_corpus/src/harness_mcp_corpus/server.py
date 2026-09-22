"""M1 corpus tools over MCP; local stdio or Streamable HTTP."""

from __future__ import annotations

import os
from pathlib import Path

from harness_corpus.store import CorpusStore
from mcp.server.fastmcp import FastMCP


def create_server(root: Path, host: str = "127.0.0.1", port: int = 8765) -> FastMCP:
    corpus = CorpusStore(root)
    mcp = FastMCP("corpus", host=host, port=port)

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

    @mcp.tool(description="Read structural facts for a paper with source fact IDs.")
    def facts(paper_id: str, offset: int = 0, limit: int = 10) -> dict:
        return corpus.facts(paper_id, offset, limit)

    @mcp.tool(
        description="Read incoming or outgoing citations with citation fact IDs. The reference title describes the cited target, not the citing paper."
    )
    def citations(paper_id: str, direction: str = "out", offset: int = 0, limit: int = 10) -> dict:
        return corpus.citations(paper_id, direction, offset, limit)

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
