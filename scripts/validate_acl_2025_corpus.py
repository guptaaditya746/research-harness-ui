"""Validate a staged ACL/EMNLP 2025 corpus before switching the live server."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from harness_corpus.store import CorpusStore
from harness_corpus.tei import paper_id


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-corpus", type=Path, required=True)
    parser.add_argument("--staged-input", type=Path, required=True)
    parser.add_argument("--built-corpus", type=Path, required=True)
    args = parser.parse_args()

    existing = rows(args.existing_corpus / "papers.jsonl")
    with (args.staged_input / "manifest.tsv").open(newline="") as handle:
        manifest = list(csv.DictReader(handle, delimiter="\t"))
    built = rows(args.built_corpus / "papers.jsonl")
    passages = rows(args.built_corpus / "passages.jsonl")
    facts = rows(args.built_corpus / "facts.jsonl")
    by_id = {paper["paper_id"]: paper for paper in built}
    expected_new = {paper_id(Path(row["filename"])) for row in manifest}
    expected_old = {paper["paper_id"] for paper in existing}
    assert len(manifest) == 300, f"expected 300 new papers, found {len(manifest)}"
    assert len(expected_new) == 300, "new paper IDs are not unique"
    assert len(expected_old) == len(existing) == 37, "existing paper set changed"
    assert len(by_id) == len(built) == 337, f"expected 337 unique papers, found {len(by_id)}"
    assert set(by_id) == expected_old | expected_new, "papers are missing or unexpected"
    assert Counter(by_id[pid]["venue"] for pid in expected_new) == {
        "ACL 2025": 150, "EMNLP 2025": 150,
    }, "proceedings counts or venues differ from the selection"

    for row in manifest:
        paper = by_id[paper_id(Path(row["filename"]))]
        for field in ("title", "doi", "venue", "anthology_id"):
            assert paper[field] == row[field], f"{row['anthology_id']} has wrong {field}"
        assert paper["year"] == 2025, f"{row['anthology_id']} has wrong year"
        assert paper["authors"] == json.loads(row["authors"]), f"{row['anthology_id']} has wrong authors"
    for old in existing:
        paper = by_id[old["paper_id"]]
        for field in (
            "title", "authors", "year", "venue", "doi", "dblp_iri", "identifier",
            "identifier_source", "dblp_status",
        ):
            assert paper[field] == old[field], f"changed existing {field}: {old['paper_id']}"

    passage_counts = Counter(item["paper_id"] for item in passages)
    assert all(passage_counts[pid] > 0 for pid in by_id), "a paper has no full-text passages"
    assert len({item["passage_id"] for item in passages}) == len(passages), "duplicate passage IDs"
    assert len({item["fact_id"] for item in facts}) == len(facts), "duplicate fact IDs"
    assert len({paper["doi"] for paper in built if paper["doi"]}) == len(
        [paper for paper in built if paper["doi"]]
    ), "duplicate DOI"

    store = CorpusStore(args.built_corpus)
    hits = store.search("language", limit=5)
    assert hits["total"] > 0, "the search index returned no hits"
    report = json.loads((args.built_corpus / "ingest_report.json").read_text())
    assert all(paper["dblp_status"] == "verified" for paper in built), "a DBLP identifier is unresolved"
    print(json.dumps({
        "papers": len(built),
        "existing_retained": len(expected_old),
        "new_papers": len(expected_new),
        "passages": len(passages),
        "facts": len(facts),
        "dblp_verified": sum(paper["dblp_status"] == "verified" for paper in built),
        "fallback_papers": len(report.get("fallback_papers", [])),
        "search_smoke_hits": hits["total"],
    }, indent=2))


if __name__ == "__main__":
    main()
