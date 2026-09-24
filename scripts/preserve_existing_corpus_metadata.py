"""Keep the original corpus's curated bibliography after a combined PDF ingest.

GROBID often omits DOI or year from an old PDF. Re-ingesting that PDF must not
erase metadata that was already verified in the existing corpus. Run only on a
completed, separate staging corpus; this command never edits the live corpus.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness_corpus.ingest import structural_facts, write_jsonl


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def preserve(existing_root: Path, staged_root: Path) -> dict:
    if existing_root.resolve() == staged_root.resolve():
        raise ValueError("the existing and staged roots must differ")
    if (staged_root / ".ingesting").exists():
        raise RuntimeError("staged ingest is still running")
    old = read_jsonl(existing_root / "papers.jsonl")
    staged = read_jsonl(staged_root / "papers.jsonl")
    old_by_id = {row["paper_id"]: row for row in old}
    if len(old_by_id) != len(old) or len({row["paper_id"] for row in staged}) != len(staged):
        raise ValueError("duplicate paper IDs")
    if not set(old_by_id) <= {row["paper_id"] for row in staged}:
        raise ValueError("staged corpus is missing an original paper")
    restored = [old_by_id.get(row["paper_id"], row) for row in staged]
    references = read_jsonl(staged_root / "references.jsonl")
    write_jsonl(staged_root / "papers.jsonl", restored)
    write_jsonl(staged_root / "facts.jsonl", structural_facts(restored, references))
    report_path = staged_root / "ingest_report.json"
    report = json.loads(report_path.read_text())
    report["unresolved_dblp"] = [row["paper_id"] for row in restored if not row.get("dblp_iri")]
    report["verified_dblp"] = sorted(row["paper_id"] for row in restored if row.get("dblp_iri"))
    report["preserved_existing_metadata"] = sorted(old_by_id)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return {
        "original_papers_preserved": len(old_by_id),
        "total_papers": len(restored),
        "verified_dblp": len(report["verified_dblp"]),
        "unresolved_dblp": len(report["unresolved_dblp"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-corpus", type=Path, required=True)
    parser.add_argument("--built-corpus", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(preserve(args.existing_corpus, args.built_corpus), indent=2))


if __name__ == "__main__":
    main()
