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
