"""Scoring runs for the approved gold: every question, every path, N repetitions, in evaluation mode.

    make benchmark-runs ARGS="--gold-version pilot-v1"   # every question of that gold, 3x on each path
    make benchmark-runs ARGS="--ids nlp11,nlp27"          # a few questions (question ids or gold row ids)
    make benchmark-runs ARGS="--paths agentic,external --repetitions 1"
    make benchmark-runs ARGS="--resume results/benchmark_runs-<time>.jsonl"   # only the missing repetitions
    make benchmark-runs ARGS="--input evidence_doc"      # the evidence document D is each run's input

Runs go through POST /benchmark/execute, exactly like "Run benchmark" on the Benchmark page: they are
stamped with the evaluation snapshot (which `make evaluate` matches against the gold rows), and an
Agentic run researches its first grounded reading and starts one run per further reading (one attempt,
scored together as a set). The question text is taken from the
gold row, so runs and gold always match. Repetitions start one after the other and at most as many
groups run at once as the API has run slots, so no run waits in the queue while its deadline runs.
Afterwards: make evaluate (gold from labelling: evidence documents, reviewed; see its README).
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import httpx
import typer
from harness_config import settings
from run_questions import summary, wait

ROOT = Path(__file__).resolve().parents[1]
ALL_PATHS = ("baseline", "baseline-enhanced", "agentic", "external")
lock = threading.Lock()


FINISHED_OK = {"answered", "corpus_absent", "no_evidence_found"}


def gold_questions(path: Path, version: str | None = None) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row for row in rows if row.get("status") == "approved"
            and (version is None or row.get("gold_version") == version)]


def question_id(row: dict[str, Any]) -> str:
    """The question id of a gold row: its question_id, else the id ("gold-pilot-v1-nlp11" -> nlp11)."""
    return str(row.get("question_id") or str(row.get("id", "")).rsplit("-", 1)[-1])


def selected(rows: list[dict[str, Any]], ids: str | None) -> list[dict[str, Any]]:
    if not ids:
        return rows
    wanted = {item.strip() for item in ids.split(",") if item.strip()}
    return [row for row in rows if question_id(row) in wanted or row.get("id") in wanted]


def main(
    gold: Annotated[Path | None, typer.Option(help="Gold rows (default: $DATASETS_ROOT/gold/candidates.jsonl).")] = None,
    gold_version: Annotated[str | None, typer.Option(help="Only gold rows of this version (e.g. pilot-v1).")] = None,
    ids: Annotated[str | None, typer.Option(help="Only these question ids, comma-separated.")] = None,
    paths: Annotated[str, typer.Option(help="Comma-separated paths (external only when configured).")] =
        ",".join(ALL_PATHS),
    repetitions: Annotated[int, typer.Option(help="Runs per question and path.")] = 3,
    budget: Annotated[str, typer.Option(help="fast or slow (Agentic always slow).")] = "slow",
    parallel: Annotated[int | None, typer.Option(help="Groups at a time (default and maximum: the API's run slots).")] = None,
    api: Annotated[str, typer.Option(help="Harness API base URL.")] = "http://127.0.0.1:8000",
    timeout: Annotated[float, typer.Option(help="Seconds to wait for one run.")] = 3600,
    resume: Annotated[Path | None, typer.Option(help="Earlier results file: skip finished groups.")] = None,
    run_input: Annotated[str | None, typer.Option("--input", help="corpus or evidence_doc (docs/RUN_INPUT.md); "
                                                                  "default: the API's HARNESS_RUN_INPUT.")] = None,
) -> None:
    source = gold or Path(settings().datasets_root or ROOT / "datasets") / "gold" / "candidates.jsonl"
    rows = selected(gold_questions(source, gold_version), ids)
    if not rows:
        raise typer.BadParameter(f"no approved gold row in {source} matches"
                                 + (f" version {gold_version}" if gold_version else "") + (f" and ids {ids}" if ids else ""))
    chosen = [item.strip() for item in paths.split(",") if item.strip()]
    unknown = [item for item in chosen if item not in ALL_PATHS]
    if unknown:
        raise typer.BadParameter(f"unknown paths: {', '.join(unknown)}")
    with httpx.Client(base_url=api, timeout=30) as client:
        external = client.get("/settings").json().get("external") == "on"
        try:
            slots = int(client.get("/models/status").json()["runs"]["capacity"])
        except (httpx.HTTPError, KeyError, TypeError, ValueError):
            slots = 2
    workers = max(1, min(parallel or slots, slots))
    if "external" in chosen and not external:
        typer.echo("External path is off in the API (HARNESS_EXTERNAL_COMMAND / _CONFIG): skipping it.")
        chosen = [item for item in chosen if item != "external"]
    finished: dict[tuple[str, str], int] = {}
    if resume and resume.exists():
        for line in resume.read_text().splitlines():
            item = json.loads(line)
            if item.get("status") in FINISHED_OK:
                key = (item["gold_id"], item["path"])
                finished[key] = finished.get(key, 0) + 1
    out = resume or ROOT / "results" / f"benchmark_runs-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    groups = [(row, path, repetitions - finished.get((row["id"], path), 0)) for row in rows for path in chosen]
    groups = [group for group in groups if group[2] > 0]
    typer.echo(f"{len(rows)} gold questions x {len(chosen)} paths: {sum(group[2] for group in groups)} runs to go, "
               f"{workers} at a time · results in {out}")

    def one(group: tuple[dict[str, Any], str, int]) -> None:
        row, path, missing = group
        base = {"question_id": question_id(row), "gold_id": row["id"], "question": row["question"], "path": path}
        with httpx.Client(base_url=api, timeout=60) as client:
            for index in range(repetitions - missing + 1, repetitions + 1):
                try:
                    response = client.post("/benchmark/execute", json={
                        "question": row["question"], "path": path, "repetitions": 1,
                        "budget": "slow" if path == "agentic" else budget,
                        **({"run_input": run_input} if run_input else {})})
                    response.raise_for_status()
                    run_id = response.json()["ids"][0]
                except httpx.HTTPError as exc:
                    record(base | {"repetition": index, "status": "error", "reason": str(exc)[:300]})
                    continue
                try:
                    record(base | {"repetition": index, "run_id": run_id, **summary(wait(client, run_id, timeout))})
                except (httpx.HTTPError, TimeoutError) as exc:
                    record(base | {"repetition": index, "run_id": run_id, "status": "error", "reason": str(exc)[:300]})

    def record(item: dict[str, Any]) -> None:
        with lock, out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        # "facts" counts the final answer's triples: an incomplete run has none, whatever passed review, so the
        # failing check (what stopped it) and the stage counts are printed with it.
        typer.echo(f"[{item['question_id']}] {item['path']} #{item.get('repetition', '-')} {item.get('run_id') or '-'} "
                   f"{item.get('status')} {item.get('facts', '')} triples in the answer"
                   + (f" ({item['reason']})" if item.get("reason") else "")
                   + (f"\n    stages: {item['stages']}" if item.get("stages") and item.get("status") != "answered" else "")
                   + (f"\n    failed: {item['failed_check']}" if item.get("failed_check") else ""))

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, groups))
    typer.echo(f"done in {(time.monotonic() - started) / 60:.1f} min. Next: make evaluate")


if __name__ == "__main__":
    typer.run(main)
