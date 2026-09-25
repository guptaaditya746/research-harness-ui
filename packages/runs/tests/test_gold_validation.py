"""Benchmark coverage: stratified by complexity level when gold carries one, else by category."""

from __future__ import annotations

import json
from pathlib import Path

from harness_runs.store import GoldCandidateStore


def _row(index: int, **extra) -> dict:
    return {"id": f"g{index}", "status": "approved", "question": f"Q{index}?",
            "gold_triples": [{"s": "p1", "p": "title", "o": "T"}], "gold_claims": [{}],
            "evidence_refs": ["x"], "gold_version": "v1", "corpus_fingerprint": "c", "ontology_hash": "h",
            "reviewer": "r", "snapshot": {"corpusSnapshotId": "c"}, **extra}


def _store(tmp_path: Path, rows: list[dict]) -> GoldCandidateStore:
    path = tmp_path / "candidates.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return GoldCandidateStore(path)


def test_complexity_levels_replace_categories_when_present(tmp_path) -> None:
    rows = [_row(i, category="synthesis", complexity={"level": level, "points": 0, "features": {}})
            for i, level in enumerate(["simple"] * 3 + ["compound"] * 3 + ["complex"] * 2)]
    report = _store(tmp_path, rows).validate_benchmark(minimum_per_category=3)
    assert report["stratified_by"] == "complexity"
    assert [error for error in report["errors"] if error["code"] == "minimum_questions"] == [
        {"category": "complex", "code": "minimum_questions", "count": 2, "required": 3}]


def test_old_gold_is_still_checked_by_category(tmp_path) -> None:
    report = _store(tmp_path, [_row(1, category="metadata")]).validate_benchmark(minimum_per_category=1)
    assert report["stratified_by"] == "category"
    assert {error["category"] for error in report["errors"]} == {"citation", "aggregation/ranking", "synthesis"}
