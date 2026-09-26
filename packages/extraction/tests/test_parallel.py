"""Parallel extraction and review batches (HARNESS_EXTRACTION_CONCURRENCY)."""

from __future__ import annotations

import json
import threading
import time
from contextvars import ContextVar

import pytest
from harness_extraction import ExtractedTriple, ExtractionEngine, SchemaInference, parallel_map
from harness_semantic import hand_written_subschema

LABEL: ContextVar[str] = ContextVar("label", default="none")


def test_parallel_map_keeps_order_and_context_and_raises_the_first_failure() -> None:
    LABEL.set("run-1")
    seen: list[str] = []

    def work(value: int) -> int:
        time.sleep(0.01 * (5 - value))          # later items finish first
        seen.append(LABEL.get())
        return value * 10

    assert parallel_map(work, [1, 2, 3, 4], workers=4) == [10, 20, 30, 40]
    assert seen == ["run-1"] * 4                 # the caller's context reaches the workers

    def fail(value: int) -> int:
        if value == 2:
            raise ValueError("batch 2")
        return value

    with pytest.raises(ValueError, match="batch 2"):
        parallel_map(fail, [1, 2, 3], workers=3)
    assert parallel_map(work, [3], workers=8) == [30]


class Reviewer:
    """Keeps every candidate whose object ends in an even digit; records how many calls overlap."""

    def __init__(self) -> None:
        self.active = 0
        self.peak = 0
        self.lock = threading.Lock()

    def complete_json(self, system: str, user: str) -> dict:
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(0.05)
        with self.lock:
            self.active -= 1
        candidates = json.loads(user)["candidates"]
        return {"keep_ids": [item["id"] for item in candidates if int(item["object"][-1]) % 2 == 0]}


def test_review_batches_run_in_parallel_and_keep_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HARNESS_EXTRACTION_CONCURRENCY", "3")
    reviewer = Reviewer()
    engine = ExtractionEngine(reviewer, hand_written_subschema())
    triples = [ExtractedTriple(subject="p4", predicate="usesDataset", object=f"data {index}",
                               source_id="p4-abstract", evidence="e") for index in range(10)]
    kept, report = engine.refine_triples("q", triples, [], batch_size=2)
    assert [item.object for item in kept] == ["data 0", "data 2", "data 4", "data 6", "data 8"]
    assert report["model_calls"] == 5 and reviewer.peak >= 2


def test_adaptive_extraction_batches_run_in_parallel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HARNESS_EXTRACTION_CONCURRENCY", "4")

    class Extractor:
        def complete_json(self, system: str, user: str) -> dict:
            passages = json.loads(user)["passages"]
            time.sleep(0.02)
            return {"triples": [{"subject": "p4", "predicate": "usesDataset", "object": row["passage_id"],
                                 "source_id": row["passage_id"], "evidence": row["text"]} for row in passages]}

    engine = ExtractionEngine(Extractor(), hand_written_subschema())
    passages = [{"passage_id": f"p4-{index}", "paper_id": "p4", "text": f"text {index}"} for index in range(8)]
    inferred = SchemaInference(label="l", gloss="g", classes=["Paper"], properties=["usesDataset"])
    triples, reports = engine.extract_batches("q", inferred, passages, batch_size=2)
    assert [item.object for item in triples] == [f"p4-{index}" for index in range(8)]
    assert [report["batch"] for report in reports] == [1, 2, 3, 4]


def test_parallel_map_starts_nothing_new_after_a_failure() -> None:
    started: list[int] = []

    def work(value: int) -> int:
        started.append(value)
        if value == 1:
            raise RuntimeError("batch 1 failed")
        time.sleep(0.2 if value == 0 else 0.01)
        return value

    with pytest.raises(RuntimeError, match="batch 1"):
        parallel_map(work, list(range(12)), workers=3)
    assert len(started) < 12                     # queued batches were cancelled, not run
