"""Shared test setup.

Stub models in the tests often return recorded replies in call order. Parallel batches would make that
order depend on thread timing, so tests run extraction batches one after another unless a test sets
HARNESS_EXTRACTION_CONCURRENCY itself (see packages/extraction/tests/test_parallel.py).
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _sequential_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HARNESS_EXTRACTION_CONCURRENCY", "1")
