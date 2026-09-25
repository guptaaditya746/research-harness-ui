from datetime import UTC, datetime

from harness_runs.analysis import parse_model_calls, stats_by_path, summarize
from harness_schema.models import Check, Run, TraceRow


def _run(run_id: str, path: str, status: str, checks: list[Check]) -> Run:
    run = Run(
        id=run_id, startedAt=datetime.now(UTC), durationS=12.5, question="Q?", path=path,
        budget="slow", status="incomplete", reason="test", checks=checks,
        trace=[
            TraceRow(atS=0.1, action="Model call started", returned="heavy-model · schema inference"),
            TraceRow(atS=3.3, action="Model call finished",
                     returned="heavy-model · schema inference · 3.2s · 640 tokens · prompt x@v2:1a2b3c4d"),
            TraceRow(atS=4.0, action="Model call failed",
                     returned="timed out after 90s (heavy-model · triple extraction)"),
        ],
    )
    # model_copy skips validation: an answered run would otherwise need readings.
    return run.model_copy(update={"status": status})


def test_model_calls_and_issues_are_read_from_the_saved_run(tmp_path) -> None:
    run = _run("r-1", "agentic", "answered", [
        Check(name="retrieval", status="pass"),
        Check(name="agent evidence IDs", status="warn", detail="1 rejected"),
        Check(name="agentic execution", status="fail", detail="degraded"),
    ])
    (tmp_path / "r-1").mkdir()
    (tmp_path / "r-1" / "tool_calls.jsonl").write_text('{"a": 1}\n{"b": 2}\n')
    summary = summarize(run, tmp_path)
    calls = parse_model_calls(run)
    assert [(c.purpose, c.seconds, c.ok) for c in calls] == [
        ("schema inference", 3.2, True), ("triple extraction", 90.0, False)]
    assert summary.tool_calls == 2 and summary.failed_model_calls == 1
    assert summary.main_issue == "agentic execution: degraded"
    assert summary.seconds_by_purpose() == {"triple extraction": 90.0, "schema inference": 3.2}


def test_stats_group_runs_by_path() -> None:
    summaries = [summarize(_run("r-1", "agentic", "answered", [])),
                 summarize(_run("r-2", "agentic", "incomplete", [Check(name="retrieval", status="fail")]))]
    [row] = stats_by_path(summaries)
    assert row["runs"] == 2 and row["answered"] == 1 and row["top_issue"] == "retrieval"
