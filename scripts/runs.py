"""Inspect saved runs quickly: what ran, how long it took, and what went wrong.

    uv run python scripts/runs.py list -n 20 [--path agentic] [--status incomplete]
    uv run python scripts/runs.py show r-1a2b3c4d
    uv run python scripts/runs.py compare r-1a2b3c4d r-5e6f7a8b
    uv run python scripts/runs.py stats -n 50
    uv run python scripts/runs.py list --fixtures        # the recorded demo runs

Reads ARTIFACT_ROOT (default ./artifacts). See docs/RUN_ANALYSIS.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from harness_config import settings
from harness_runs.analysis import RunSummary, load_runs, stats_by_path, summarize
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__.split("\n\n")[0])
console = Console()
Fixtures = Annotated[bool, typer.Option("--fixtures", help="Read the recorded demo runs instead.")]


def _root(fixtures: bool) -> Path:
    if fixtures:
        return settings().fixtures_dir or ROOT / "web" / "fixtures"
    return settings().artifact_root or ROOT / "artifacts"


def _summaries(fixtures: bool) -> tuple[Path, list[RunSummary]]:
    root = _root(fixtures)
    return root, [summarize(run, root) for run in load_runs(root)]


def _find(run_id: str, fixtures: bool) -> RunSummary:
    root, summaries = _summaries(fixtures)
    for item in summaries:
        if item.run.id == run_id:
            return item
    console.print(f"[red]No run {run_id} under {root}[/red]")
    raise typer.Exit(1)


def _status(status: str) -> str:
    colour = {"answered": "green", "incomplete": "red", "stopped": "yellow"}.get(status, "cyan")
    return f"[{colour}]{status}[/{colour}]"


@app.command("list")
def list_runs(
    n: Annotated[int, typer.Option("-n", help="How many runs.")] = 20,
    path: Annotated[str | None, typer.Option(help="agentic, baseline or baseline-enhanced.")] = None,
    status: Annotated[str | None, typer.Option(help="answered, incomplete, ...")] = None,
    fixtures: Fixtures = False,
    as_json: Annotated[bool, typer.Option("--json", help="One JSON object per run.")] = False,
) -> None:
    """Latest runs: path, status, time, model and tool calls, and the main issue."""
    root, summaries = _summaries(fixtures)
    rows = [s for s in summaries if (not path or s.run.path == path) and (not status or s.run.status == status)]
    if as_json:
        for item in rows[:n]:
            run = item.run
            console.print_json(json.dumps({
                "id": run.id, "started": run.started_at.isoformat(), "path": run.path,
                "status": run.status, "reason": run.reason, "duration_s": run.duration_s,
                "model_calls": len(item.model_calls), "failed_model_calls": item.failed_model_calls,
                "model_seconds": item.model_seconds, "tool_calls": item.tool_calls,
                "main_issue": item.main_issue, "question": run.question,
            }))
        return
    table = Table(title=f"Runs in {root}", show_lines=False)
    for column, width in (("id", 10), ("started", 11), ("path", 17), ("status", 16), ("time s", 6),
                          ("model s", 7), ("calls", 5), ("tools", 5)):
        table.add_column(column, no_wrap=True, min_width=width)
    table.add_column("main issue", max_width=48, overflow="ellipsis", no_wrap=True)
    table.add_column("question", max_width=40, overflow="ellipsis", no_wrap=True)
    for item in rows[:n]:
        run = item.run
        table.add_row(
            run.id, run.started_at.strftime("%m-%d %H:%M"), run.path, _status(run.status),
            f"{run.duration_s:.0f}", f"{item.model_seconds:.0f}",
            f"{len(item.model_calls)}" + (f" ({item.failed_model_calls} failed)" if item.failed_model_calls else ""),
            str(item.tool_calls), item.main_issue or run.reason or "", run.question,
        )
    console.print(table)


@app.command()
def show(run_id: str, fixtures: Fixtures = False) -> None:
    """One run: checks, where the time went, slowest model calls, prompts used."""
    item = _find(run_id, fixtures)
    run = item.run
    console.print(f"[bold]{run.id}[/bold] · {run.path} · {run.budget} · {_status(run.status)}"
                  f" · {run.duration_s:.1f}s" + (f" · reason: {run.reason}" if run.reason else ""))
    console.print(f"Question: {run.question}")
    if run.concept:
        console.print(f"Concept: {run.concept.label}")
    checks = Table(title="Checks")
    for column in ("check", "status", "detail"):
        checks.add_column(column, overflow="fold")
    for check in run.checks:
        checks.add_row(check.name, check.status, check.detail)
    console.print(checks)
    purposes = Table(title=f"Model time by purpose ({item.model_seconds:.1f}s in {len(item.model_calls)} calls)")
    purposes.add_column("purpose")
    purposes.add_column("seconds", justify="right")
    for purpose, seconds in item.seconds_by_purpose().items():
        purposes.add_row(purpose or "?", f"{seconds:.1f}")
    console.print(purposes)
    slow = Table(title="Slowest model calls")
    for column in ("seconds", "model", "purpose", "ok"):
        slow.add_column(column)
    for call in item.slowest_calls():
        slow.add_row(f"{call.seconds:.1f}", call.model, call.purpose, "yes" if call.ok else "[red]no[/red]")
    console.print(slow)
    if run.prompt_hashes:
        console.print("Prompts: " + ", ".join(f"{name} {version}" for name, version in sorted(run.prompt_hashes.items())))
    if run.chosen_reading is not None and run.readings:
        console.print(f"\n[bold]Answer[/bold]: {run.readings[run.chosen_reading].answer}")


@app.command()
def compare(first: str, second: str, fixtures: Fixtures = False) -> None:
    """Two runs side by side (for example before and after a change)."""
    a, b = _find(first, fixtures), _find(second, fixtures)
    table = Table(title="Comparison")
    for column in ("", a.run.id, b.run.id):
        table.add_column(column, overflow="fold")
    rows = [
        ("path", a.run.path, b.run.path),
        ("status", a.run.status, b.run.status),
        ("duration s", f"{a.run.duration_s:.1f}", f"{b.run.duration_s:.1f}"),
        ("model calls", str(len(a.model_calls)), str(len(b.model_calls))),
        ("model seconds", f"{a.model_seconds:.1f}", f"{b.model_seconds:.1f}"),
        ("tool calls", str(a.tool_calls), str(b.tool_calls)),
        ("main issue", a.main_issue, b.main_issue),
        ("answer triples", str(_triples(a)), str(_triples(b))),
    ]
    checks = sorted({c.name for c in a.run.checks} | {c.name for c in b.run.checks})
    status_a = {c.name: c.status for c in a.run.checks}
    status_b = {c.name: c.status for c in b.run.checks}
    rows += [(f"check: {name}", status_a.get(name, "-"), status_b.get(name, "-")) for name in checks
             if status_a.get(name) != status_b.get(name)]
    for row in rows:
        table.add_row(*row)
    console.print(table)


def _triples(item: RunSummary) -> int:
    run = item.run
    return len(run.readings[run.chosen_reading].triples) if run.chosen_reading is not None and run.readings else 0


@app.command()
def stats(n: Annotated[int, typer.Option("-n", help="Runs to include.")] = 50, fixtures: Fixtures = False) -> None:
    """Per path over the latest runs: answered share, median time and calls, top issue."""
    _, summaries = _summaries(fixtures)
    table = Table(title=f"Latest {min(n, len(summaries))} runs by path")
    columns = ("path", "runs", "answered", "median_duration_s", "median_model_calls", "median_model_seconds", "top_issue")
    for column in columns:
        table.add_column(column)
    for row in stats_by_path(summaries[:n]):
        table.add_row(*(str(row[column]) for column in columns))
    console.print(table)


if __name__ == "__main__":
    app()
