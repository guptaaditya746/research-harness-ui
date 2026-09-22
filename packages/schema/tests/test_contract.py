from datetime import UTC, datetime

import pytest
from harness_schema.models import Metrics, Run, required_artifacts, terminal_scene
from pydantic import ValidationError


def sample(status: str, path: str = "agentic", reason: str | None = None) -> Run:
    return Run.model_validate(
        {
            "id": "r-test",
            "startedAt": datetime.now(UTC).isoformat(),
            "durationS": 1,
            "question": "What?",
            "path": path,
            "budget": "fast",
            "status": status,
            "reason": reason,
            "readings": []
            if status != "answered"
            else [
                {
                    "label": "One",
                    "gloss": "One",
                    "classes": [],
                    "axioms": [],
                    "sparql": "",
                    "answer": "A",
                    "triples": [],
                    "metrics": Metrics().model_dump(),
                }
            ],
            "chosenReading": 0 if status == "answered" else None,
        }
    )


@pytest.mark.parametrize(
    "status,scene",
    [
        ("answered", "agentic_result"),
        ("awaiting_reading", "readings"),
        ("incomplete", "failed"),
        ("stopped", "idle"),
    ],
)
def test_terminal_state_scene(status: str, scene: str) -> None:
    assert (
        terminal_scene(sample(status, reason="no_results" if status == "incomplete" else None))
        == scene
    )


def test_incomplete_requires_reason() -> None:
    with pytest.raises(ValidationError):
        sample("incomplete")


def test_artifact_contract_per_path() -> None:
    assert "run.json" in required_artifacts("agentic")
    assert "verification.json" not in required_artifacts("baseline")
    assert len(required_artifacts("agentic")) == 17
