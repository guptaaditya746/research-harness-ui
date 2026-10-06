"""The Settings page's API: the editable settings, their validation, persistence, and what runs record."""

import json
import time
from pathlib import Path

import app_state
from fastapi.testclient import TestClient
from main import RunRuntime, app
from test_api import FakeRunner


def _runtime(tmp_path: Path, monkeypatch) -> RunRuntime:
    monkeypatch.setenv("HARNESS_UI_SETTINGS_FILE", str(tmp_path / "ui.json"))
    # Restored after the test: the Settings page writes the process environment.
    monkeypatch.setenv("HARNESS_EVIDENCE_DOC_BATCH", "24")
    monkeypatch.setenv("HARNESS_RUN_INPUT", "corpus")
    monkeypatch.setenv("HARNESS_EVIDENCE_DOC_ROOT", str(tmp_path / "docs"))
    runtime = RunRuntime(tmp_path / "artifacts", tmp_path / "datasets", runner_factory=lambda sink: FakeRunner(sink))
    monkeypatch.setattr(app_state, "runtime", runtime)
    return runtime


def test_settings_are_listed_validated_saved_and_applied_again(tmp_path: Path, monkeypatch) -> None:
    runtime = _runtime(tmp_path, monkeypatch)
    client = TestClient(app)
    listed = {row["name"]: row for row in client.get("/config").json()["settings"]}
    assert listed["HARNESS_RUN_INPUT"]["options"] == ["corpus", "evidence_doc"]
    assert listed["HARNESS_EVIDENCE_DOC_BATCH"]["min"] == 4 and listed["HARNESS_EVIDENCE_DOC_BATCH"]["value"] == "24"
    changed = client.put("/config", json={"values": {"HARNESS_EVIDENCE_DOC_BATCH": "40"}})
    assert changed.status_code == 200 and changed.json()["overrides"] == {"HARNESS_EVIDENCE_DOC_BATCH": "40"}
    assert json.loads((tmp_path / "ui.json").read_text()) == {"HARNESS_EVIDENCE_DOC_BATCH": "40"}
    assert client.put("/config", json={"values": {"HARNESS_EVIDENCE_DOC_BATCH": "1"}}).status_code == 422
    assert client.put("/config", json={"values": {"VLLM_API_KEY": "x"}}).status_code == 422
    runtime.running.add("r-busy")
    assert client.put("/config", json={"values": {"HARNESS_EVIDENCE_DOC_BATCH": "30"}}).status_code == 409
    runtime.running.discard("r-busy")
    # A restarted API applies the saved changes again; null restores the earlier value.
    monkeypatch.setenv("HARNESS_EVIDENCE_DOC_BATCH", "24")
    again = RunRuntime(tmp_path / "artifacts", tmp_path / "datasets", runner_factory=lambda sink: FakeRunner(sink))
    assert again.ui_overrides == {"HARNESS_EVIDENCE_DOC_BATCH": "40"}
    monkeypatch.setattr(app_state, "runtime", again)
    restored = client.put("/config", json={"values": {"HARNESS_EVIDENCE_DOC_BATCH": None}}).json()
    assert restored["overrides"] == {}
    assert {row["name"]: row["value"] for row in restored["settings"]}["HARNESS_EVIDENCE_DOC_BATCH"] == "24"


def test_a_run_records_its_settings_and_input(tmp_path: Path, monkeypatch) -> None:
    _runtime(tmp_path, monkeypatch)
    client = TestClient(app)
    client.put("/config", json={"values": {"HARNESS_EVIDENCE_DOC_BATCH": "40"}})
    run_id = client.post("/runs", json={"question": "Which paper introduced VPP?", "path": "baseline",
                                        "budget": "fast"}).json()["id"]
    for _ in range(50):
        run = client.get(f"/runs/{run_id}").json()
        if run.get("settings"):
            break
        time.sleep(0.05)
    assert run["runInput"] == "corpus" and run["settings"]["HARNESS_EVIDENCE_DOC_BATCH"] == "40"


def test_d_as_input_needs_the_questions_document(tmp_path: Path, monkeypatch) -> None:
    _runtime(tmp_path, monkeypatch)
    client = TestClient(app)
    refused = client.post("/runs", json={"question": "Which paper introduced VPP?", "path": "baseline",
                                         "budget": "fast", "run_input": "evidence_doc"})
    assert refused.status_code == 422 and "no evidence document" in refused.text
    assert client.post("/runs", json={"question": "q", "path": "baseline", "budget": "fast",
                                      "run_input": "papers"}).status_code == 422
