"""Evaluation mode: one unattended agentic attempt researches every proposed reading."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import main
import runtime as runtime_module
from harness_schema.models import ConceptChoice, Metrics, Reading, Run, required_artifacts
from main import RunRuntime


def _proposal(run_id: str, labels: list[str]) -> dict:
    readings = [Reading(label=label, gloss="g", classes=[], axioms=[], sparql="", answer="", triples=[],
                        metrics=Metrics(), status="proposed") for label in labels]
    run = Run(id=run_id, startedAt=datetime.now(UTC), durationS=0.0, question="Q about MethodX?", path="agentic",
              budget="fast", status="awaiting_reading", readings=readings,
              concepts=[ConceptChoice(label=label, gloss="g", evidenceHits=2) for label in labels],
              artifacts=list(required_artifacts("agentic")))
    return {"run": run, "artifacts": {"plans.json": b"[]"}}


def _execute(tmp_path: Path, monkeypatch, labels: list[str]) -> tuple[dict, list, dict]:
    test_runtime = RunRuntime(tmp_path / "artifacts", tmp_path / "datasets")
    started: list = []
    researched: dict = {}

    class FakeModel:
        def __init__(self, model: str, timeout_s: float) -> None:
            self.model = model

    class FakeAgenticResearch:
        def __init__(self, baseline, sink, agent_model, high_model) -> None:
            pass

        def propose(self, question: str, budget: str, run_id: str) -> dict:
            return _proposal(run_id, labels)

        def run(self, question: str, budget: str, run_id: str, concept=None) -> dict:
            researched["concept"] = concept
            reading = Reading(label=concept.label, gloss="g", classes=[], axioms=[], sparql="", answer="a",
                              triples=[], metrics=Metrics(), status="answered")
            run = Run(id=run_id, startedAt=datetime.now(UTC), durationS=1.0, question=question, path="agentic",
                      budget=budget, status="answered", readings=[reading], chosenReading=0, concept=concept,
                      artifacts=list(required_artifacts("agentic")))
            return {"run": run, "artifacts": {}}

    monkeypatch.setattr(runtime_module, "OpenAICompatibleModel", FakeModel)
    monkeypatch.setattr(runtime_module, "AgenticResearch", FakeAgenticResearch)
    monkeypatch.setattr(test_runtime, "_runner", lambda sink, enhanced=False, **kwargs: object())
    monkeypatch.setattr(test_runtime, "start", lambda request: started.append(request) or f"r-child{len(started)}")
    request = main.RunRequest(question="Q about MethodX?", path="agentic", budget="fast", evaluation_mode=True)
    result = test_runtime._execute(request, "Q about MethodX?", "r-eval", lambda _: None, lambda *_: [])
    return result, started, researched


def test_an_evaluation_attempt_researches_every_reading_under_one_attempt_id(tmp_path: Path, monkeypatch) -> None:
    result, started, researched = _execute(tmp_path, monkeypatch, ["Most papers", "Most cited", "Most venues"])
    assert researched["concept"].label == "Most papers" and researched["concept"].auto_selected
    assert result["run"].attempt_id == "r-eval"
    assert [request.concept.label for request in started] == ["Most cited", "Most venues"]
    assert {request.attempt_id for request in started} == {"r-eval"}
    assert all(request.parent_run_id == "r-eval" and request.evaluation_mode for request in started)
    assert all(request.concept.auto_selected for request in started)


def test_a_single_reading_is_one_run_without_an_attempt_id(tmp_path: Path, monkeypatch) -> None:
    result, started, researched = _execute(tmp_path, monkeypatch, ["Only reading"])
    assert researched["concept"].label == "Only reading"
    assert started == [] and result["run"].attempt_id is None
