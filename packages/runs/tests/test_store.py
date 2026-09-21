from pathlib import Path

import pytest
from harness_runs import ArtifactStore, invariant_failure
from harness_schema.generate import fixture_runs
from harness_schema.models import required_artifacts


@pytest.mark.parametrize("fixture_index", range(4))
def test_terminal_run_gets_full_artifact_set(tmp_path: Path, fixture_index: int) -> None:
    store = ArtifactStore(tmp_path)
    run = fixture_runs()[fixture_index]
    store.finalize(run, {"answer.md": b"sample answer"})
    assert store.load_run(run.id) == run
    assert (tmp_path / run.id / "answer.md").read_bytes() == b"sample answer"
    assert all((tmp_path / run.id / name).exists() for name in required_artifacts(run.path))
    assert (tmp_path / run.id / "ledger.json").read_bytes() == b""


def test_stopped_run_gets_full_artifact_set(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    run = fixture_runs()[0].model_copy(
        update={"id": "r-stopped", "status": "stopped", "chosen_reading": None}
    )
    store.finalize(run, {})
    assert store.load_run(run.id) == run
    assert all((tmp_path / run.id / name).exists() for name in required_artifacts(run.path))


@pytest.mark.parametrize("number", range(2, 7))
def test_invariant_violation_ends_run_in_named_terminal_state(number: int) -> None:
    failed = invariant_failure(fixture_runs()[2], number)
    assert failed.status == "incomplete"
    assert failed.reason == f"invariant:{number}"
