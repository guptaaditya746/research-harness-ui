"""How much of the corpus each question touches, before running it (read-only, no models needed).

    uv run python scripts/question_coverage.py                                  # the new questions (nlp32-61)
    uv run python scripts/question_coverage.py --questions datasets/question_library_2025.json
    uv run python scripts/question_coverage.py --ids nlp32,nlp44 --show 10

For each question: the distinct papers among the top passages of a BM25 search for the question
(plus dense embeddings when HARNESS_RETRIEVAL_MODE=hybrid and the index exists), and the titles of the
best-ranked papers. A corpus-wide question whose top papers are off-topic, or that reaches only one or
two papers, should be reworded or replaced before gold is built. Questions answered from DBLP facts
(metadata, citation, aggregation) are checked by `labelling validate` instead, which computes their gold.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from harness_config import settings
from harness_corpus.store import CorpusStore

ROOT = Path(__file__).resolve().parents[1]
FACT_CATEGORIES = {"metadata", "citation", "aggregation"}


def main(
    questions: Annotated[Path, typer.Option(help="Question set (JSON list with id and question).")] =
        ROOT / "datasets" / "question_library_2025_b.json",
    ids: Annotated[str | None, typer.Option(help="Only these question ids, comma-separated.")] = None,
    depth: Annotated[int, typer.Option(help="Top passages to look at per question.")] = 40,
    show: Annotated[int, typer.Option(help="Best-ranked paper titles to print per question.")] = 5,
) -> None:
    corpus = CorpusStore(Path(settings().corpus_root or ROOT / "corpus_data"))
    rows = json.loads(questions.read_text(encoding="utf-8"))
    wanted = {part.strip() for part in ids.split(",")} if ids else None
    typer.echo(f"{len(corpus.papers)} papers, {len(corpus.passages)} passages; top {depth} passages per question\n")
    for row in rows:
        if wanted is not None and row["id"] not in wanted:
            continue
        if row.get("category") in FACT_CATEGORIES:
            typer.echo(f"{row['id']}  (answered from DBLP facts: check with `labelling validate`)  {row['question']}\n")
            continue
        items = corpus.hybrid_rank(row["question"], limit=depth)["items"]
        papers: list[str] = []
        for item in items:
            if str(item["paper_id"]) not in papers:
                papers.append(str(item["paper_id"]))
        typer.echo(f"{row['id']}  {len(papers)} papers in the top {len(items)} passages  {row['question']}")
        for paper in papers[:show]:
            typer.echo(f"    {paper}: {(corpus.paper_title(paper) or '')[:100]}")
        typer.echo("")


if __name__ == "__main__":
    typer.run(main)
