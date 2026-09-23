from pathlib import Path

from harness_runs.relations import RelationRegistry


def test_dynamic_relation_starts_provisional_and_reuses_normalized_label(tmp_path: Path) -> None:
    registry = RelationRegistry(tmp_path / "relations.sqlite")
    created = registry.propose("evaluatedInEnvironment", "Reports evaluation in an environment.", run_id="r-1")
    reused = registry.propose("Evaluated in environment", "ignored duplicate definition", run_id="r-2")
    assert created["status"] == "provisional"
    assert reused["reused"] is True
    assert reused["id"] == created["id"]
    assert registry.review(created["id"], "approved")["status"] == "approved"
