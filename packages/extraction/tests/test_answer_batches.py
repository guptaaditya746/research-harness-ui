"""Evidence that does not fit one answer prompt is answered in batches that fit, each keeping its sources."""

import json
import threading

import pytest
from harness_extraction import (
    ContextOverflow,
    ExtractedTriple,
    ExtractionEngine,
    SchemaInference,
    UnsupportedAnswer,
)
from harness_semantic import hand_written_subschema

INFERRED = SchemaInference(label="Benchmarks", gloss="g", classes=["Paper"], properties=["evaluatesOn"])


def triples(n: int) -> list[ExtractedTriple]:
    return [ExtractedTriple(subject=f"p{i}", predicate="evaluatesOn", object=f"Bench{i}", source_id=f"p{i}-s1",
                            evidence=f"MethodX{i} is evaluated on the Bench{i} benchmark " + "x" * 200)
            for i in range(n)]


class Composer:
    """Answers from whatever triples it is given, citing each; verification accepts every sentence."""

    model = "heavy-model"

    def __init__(self, unsupported: set[str] = frozenset(), overflow_above: int = 10**9) -> None:
        self.prompts: list[int] = []
        self.unsupported, self.overflow_above = unsupported, overflow_above
        self.lock = threading.Lock()

    def complete_json(self, system: str, user: str) -> dict:
        payload = json.loads(user)
        items = payload["triples"]
        with self.lock:
            self.prompts.append(len(system) + len(user))
        if len(items) > self.overflow_above:
            raise ContextOverflow("context_window_exceeded: too big")
        sentences = [f"{t['subject']} evaluates on {t['object']} [{t['source_id']}]." for t in items]
        if "draft_answer" in payload:
            bad = any(t["source_id"] in self.unsupported for t in items)
            return {"supported": not bad, "answer": payload["draft_answer"], "removed_claims": [],
                    "sentences": [{"text": text, "supported": not bad, "source_ids": [t["source_id"]]}
                                  for text, t in zip(sentences, items, strict=True)]}
        return {"answer": " ".join(sentences), "cited_source_ids": [t["source_id"] for t in items]}


def test_small_evidence_is_one_call_as_before(monkeypatch) -> None:
    model = Composer()
    report: dict = {}
    result = ExtractionEngine(model, hand_written_subschema()).answer("q", INFERRED, triples(3), report=report)
    assert len(model.prompts) == 1 and "batches" not in report and len(result.cited_source_ids) == 3


def test_large_evidence_is_split_into_prompts_that_fit_and_joined(monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_MODEL_CONTEXT_TOKENS", "6000")
    monkeypatch.setenv("HARNESS_MAX_OUTPUT_TOKENS", "1000")
    model = Composer()
    report: dict = {}
    evidence = triples(60)
    result = ExtractionEngine(model, hand_written_subschema()).answer("q", INFERRED, evidence, verify=True,
                                                                      report=report)
    assert report["batches"]["composed"] > 1 and report["batches"]["unsupported"] == 0
    assert result.cited_source_ids == [t.source_id for t in evidence]          # every batch, in order
    assert len(result.sentences) == 60 and all(item["verdict"] == "accepted" for item in result.sentences)
    assert max(model.prompts) / 3.0 < 6000 - 1000                               # each prompt leaves room to reply


def test_an_unsupported_batch_is_left_out_and_all_unsupported_fails(monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_MODEL_CONTEXT_TOKENS", "6000")
    monkeypatch.setenv("HARNESS_MAX_OUTPUT_TOKENS", "1000")
    evidence = triples(60)
    report: dict = {}
    model = Composer(unsupported={"p0-s1"})
    result = ExtractionEngine(model, hand_written_subschema()).answer("q", INFERRED, evidence, verify=True,
                                                                      report=report)
    assert report["batches"]["unsupported"] == 1 and "p0-s1" not in result.cited_source_ids
    with pytest.raises(UnsupportedAnswer, match="no batch"):
        ExtractionEngine(Composer(unsupported={t.source_id for t in evidence}), hand_written_subschema()).answer(
            "q", INFERRED, evidence, verify=True)


def test_a_cap_spreads_the_batches_and_an_overflow_halves_the_evidence(monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_MODEL_CONTEXT_TOKENS", "6000")
    monkeypatch.setenv("HARNESS_MAX_OUTPUT_TOKENS", "1000")
    monkeypatch.setenv("HARNESS_ANSWER_MAX_BATCHES", "2")
    report: dict = {}
    result = ExtractionEngine(Composer(), hand_written_subschema()).answer("q", INFERRED, triples(60), report=report)
    assert report["batches"]["composed"] == 2 < report["batches"]["total"]
    assert "p0-s1" in result.cited_source_ids and len(result.cited_source_ids) < 60
    monkeypatch.delenv("HARNESS_ANSWER_MAX_BATCHES")
    monkeypatch.setenv("HARNESS_MODEL_CONTEXT_TOKENS", "32768")
    model = Composer(overflow_above=4)                              # the estimate was too low: halve and retry
    result = ExtractionEngine(model, hand_written_subschema()).answer("q", INFERRED, triples(12))
    assert len(result.cited_source_ids) == 12


def test_evidence_review_batches_fit_the_context_whatever_the_batch_size(monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_MODEL_CONTEXT_TOKENS", "6000")
    monkeypatch.setenv("HARNESS_MAX_OUTPUT_TOKENS", "1000")
    sizes: list[int] = []

    class Reviewer:
        model = "heavy-model"

        def complete_json(self, system: str, user: str) -> dict:
            sizes.append(len(system) + len(user))
            return {"keep_ids": list(range(len(json.loads(user)["candidates"]))), "drop_reasons": {}}

    evidence = triples(60)
    passages = [{"passage_id": t.source_id, "text": "y" * 1500} for t in evidence]
    kept, report = ExtractionEngine(Reviewer(), hand_written_subschema()).refine_triples(
        "q", evidence, passages, batch_size=200)
    assert len(kept) == 60 and report["model_calls"] > 1 and max(sizes) / 3.0 < 6000 - 1000
