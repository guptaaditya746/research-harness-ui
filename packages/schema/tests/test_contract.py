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
            "evidenceSurface": "indexed_full_text" if status == "no_evidence_found" else None,
            "searchCoverage": {"coverage_complete": False} if status == "no_evidence_found" else None,
        }
    )


@pytest.mark.parametrize(
    "status,scene",
    [
        ("answered", "agentic_result"),
        ("no_evidence_found", "agentic_result"),
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
    assert "claims.json" in required_artifacts("agentic")
    assert len(required_artifacts("agentic")) == 23
    # The agent's (enhanced) pipeline reports; Fast has none of them, Adaptive all.
    assert "extraction_batches.json" in required_artifacts("agentic")
    assert "extraction_batches.json" not in required_artifacts("baseline")
    assert "extraction_batches.json" in required_artifacts("baseline-enhanced")
    assert len(set(required_artifacts("baseline-enhanced"))) == len(required_artifacts("baseline-enhanced"))
