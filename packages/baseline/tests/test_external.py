"""The External path: the external harness is a subprocess; here a small script plays its part."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from harness_baseline.external import ExternalHarness, predicate_name
from harness_corpus.store import CorpusStore
from harness_extraction import ExtractionEngine, ModelFailure
from harness_semantic import hand_written_subschema

FAKE = '''
import json, os, sys, time
from pathlib import Path
args = sys.argv[1:]
out = Path(args[args.index("--out") + 1]); out.mkdir(parents=True, exist_ok=True)
question = Path(args[args.index("--question-file") + 1]).read_text()
mode = Path(__file__).with_name("mode.txt").read_text().strip()
if mode == "crash":
    print("Traceback: boom", file=sys.stderr); sys.exit(3)
if mode == "hang":
    time.sleep(30)
events = [{"type": "start"}, {"type": "model_call", "ok": True, "seconds": 1.5, "model": "openai/heavy-model",
          "prompt_tokens": 900, "completion_tokens": 100},
          {"type": "command", "command": "kbsearch video prediction"}, {"type": "finish", "status": "answered", "triples": 2}]
for event in events:
    print(json.dumps(event), flush=True)
print("not json from a library")
triples = [
    {"subject": "p4", "predicate": "uses dataset", "object": "robot videos", "source_id": "p4-abstract",
     "evidence": "robot learning"},
    {"subject": "p4", "predicate": "dblp:authoredBy", "object": "Ada Byron", "source_id": "p4-metadata", "evidence": ""},
    {"subject": "Grace Hopper", "predicate": "dblp:authorOf", "object": "p4", "source_id": "p4-metadata", "evidence": ""},
    {"subject": "p4", "predicate": "rdf:type", "object": "Paper", "source_id": "p4-metadata", "evidence": ""},
    {"subject": "robot videos", "predicate": "cskg-ont:usedBy", "object": "MethodX",
     "source_id": "p4-abstract", "evidence": "robot learning"},
]
status = {"answered": "answered", "empty": "answered", "limits": "limits_exceeded", "other_corpus": "answered",
          "provider": "error"}[mode]
corpus = "/somewhere/else" if mode == "other_corpus" else os.environ.get("CORPUS_ROOT", "")
result = {"status": status, "answer": f"Answer to: {question} [p4-abstract].", "triples": [] if mode == "empty" else triples,
          "dropped_triples": [{"triple": {}, "why": "unknown source_id 'x'"}], "exit_status": "Submitted",
          "problem": "answer.json is missing" if mode == "limits" else None,
          "usage": {"model_calls": 1, "commands": 1}, "duration_s": 2.0,
          "harness": {"name": "mini-swe-agent", "version": "2.4.6"}, "config": {"model": "openai/heavy-model"},
          "error": "APIConnectionError: connection refused" if mode == "provider" else None,
          "knowledge_base": {"corpus_dir": corpus, "passages": 1}, "evidence_not_in_source": 1,
          "env": sorted(os.environ), "cwd_tmp": str(out.parent), "events": events}
(out / "result.json").write_text(json.dumps(result))
(out / "trajectory.json").write_text(json.dumps({"messages": []}))
'''


class NoModel:
    model = "none"

    def complete_json(self, system: str, user: str) -> dict:
        raise ModelFailure("no harness model on the External path")


def corpus(tmp_path: Path) -> CorpusStore:
    rows = {
        "papers.jsonl": [{"paper_id": "p4", "title": "Video Prediction Policy", "year": 2025}],
        "passages.jsonl": [{"passage_id": "p4-abstract", "paper_id": "p4", "section_path": "Abstract", "para_idx": 0,
                            "text": "We propose the Video Prediction Policy for robot learning."}],
        "facts.jsonl": [{"fact_id": "p4-metadata-0", "subject": "p4", "predicate": "title",
                         "object": "Video Prediction Policy", "source_id": "p4-metadata"}],
        "references.jsonl": [],
    }
    folder = tmp_path / "corpus"
    folder.mkdir()
    for name, values in rows.items():
        (folder / name).write_text("".join(json.dumps(row) + "\n" for row in values))
    return CorpusStore(folder)


def runner(tmp_path: Path, mode: str, rows: list | None = None) -> ExternalHarness:
    script = tmp_path / "fake_xh.py"
    script.write_text(FAKE)
    (tmp_path / "mode.txt").write_text(mode)
    return ExternalHarness(corpus(tmp_path), ExtractionEngine(NoModel(), hand_written_subschema()),
                           trace_sink=(rows.append if rows is not None else None),
                           command=f"{sys.executable} {script}", config_path=str(tmp_path / "config.yaml"))


def test_predicate_names_become_harness_properties() -> None:
    assert predicate_name("uses dataset") == "usesDataset"
    assert predicate_name("hrn:usesDataset") == "usesDataset"
    assert predicate_name("https://dblp.org/rdf/schema#yearOfPublication") == "year"
    assert predicate_name("author") == "authoredBy"
    assert predicate_name("  !! ") is None


def test_an_external_run_becomes_a_normal_answered_run(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DATASETS_ROOT", "/gold/lives/here")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret")
    rows: list = []
    result = runner(tmp_path, "answered", rows).run("Which data does VPP use?", run_id="r-ext1")
    run, artifacts = result["run"], result["artifacts"]
    assert run.path == "external" and run.status == "answered" and run.model_agent == "openai/heavy-model"
    reading = run.readings[0]
    assert {(t.p, t.o) for t in reading.triples} == {("usesDataset", "robot videos"), ("authoredBy", "Ada Byron"),
                                                    ("authoredBy", "Grace Hopper"),      # authorOf inverted, type dropped
                                                    ("uses", "robot videos")}            # CS-KG usedBy inverted
    uses = next(t for t in reading.triples if t.p == "uses")
    assert uses.s.endswith("MethodX")
    assert reading.answer == "Answer to: Which data does VPP use? [p4-abstract]."      # kept as the agent wrote it
    actions = [row.action for row in rows]
    assert actions[:4] == ["Starting external harness", "Model call finished", "External command",
                           "External harness finished"]
    assert rows[1].returned == "heavy-model · external agent · 1.5s · 1000 tokens"
    checks = {check.name: check for check in run.checks}
    detail = checks["external harness"].detail
    assert checks["external harness"].status == "warn" and "1 dropped by the answer contract" in detail
    assert "1 with a predicate that is not a relation" in detail and "1 quote evidence not found" in detail
    # The process saw neither our keys nor our data paths, got an absolute corpus path, and its folder is gone.
    external = json.loads(artifacts["external_result.json"])
    assert "DATASETS_ROOT" not in external["env"] and "LANGFUSE_SECRET_KEY" not in external["env"]
    assert "CORPUS_ROOT" in external["env"] and Path(external["knowledge_base"]["corpus_dir"]).is_absolute()
    assert not Path(external["cwd_tmp"]).exists()
    assert checks["retrieval"].status == "skipped"
    assert {"external_result.json", "external_trajectory.json", "triples.ttl"} <= set(artifacts)
    assert json.loads(artifacts["tool_calls.jsonl"].decode().splitlines()[0])["tool"] == "kbsearch"


@pytest.mark.parametrize(("mode", "reason"), [("crash", "external_failed"), ("limits", "external_limits_exceeded"),
                                              ("empty", "no_results"), ("other_corpus", "external_failed"),
                                              ("provider", "provider_error")])
def test_external_failures_end_incomplete_with_a_reason(tmp_path: Path, mode: str, reason: str) -> None:
    run = runner(tmp_path, mode).run("q", run_id="r-ext2")["run"]
    assert run.status == "incomplete" and run.reason == reason
    failed = [check for check in run.checks if check.status == "fail"]
    assert failed and failed[0].name == "external harness"
    if mode == "crash":
        assert "exit code 3" in failed[0].detail and "boom" in failed[0].detail
    if mode == "other_corpus":
        assert "read the corpus in /somewhere/else" in failed[0].detail
    if mode == "empty":
        assert "Answer to: q" in failed[0].detail                   # the agent's own words are kept


def test_an_unconfigured_external_path_says_so(tmp_path: Path) -> None:
    harness = ExternalHarness(corpus(tmp_path), ExtractionEngine(NoModel(), hand_written_subschema()),
                              command="", config_path="")
    run = harness.run("q")["run"]
    assert run.reason == "external_failed" and "HARNESS_EXTERNAL_COMMAND" in run.checks[-1].detail


def test_stop_kills_the_external_process(tmp_path: Path) -> None:
    import time

    from harness_extraction import call_budget

    harness = runner(tmp_path, "hang")
    started = time.monotonic()
    with call_budget(time.monotonic() + 1.5, lambda: False):
        run = harness.run("q", run_id="r-ext3")["run"]
    assert run.status == "incomplete" and run.reason == "provider_error"
    assert time.monotonic() - started < 10
