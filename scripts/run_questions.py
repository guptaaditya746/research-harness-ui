"""Run the evaluation questions through a running harness API; agentic runs cover every proposed reading.

    make run-questions                                   # the 30 questions, Agentic, every reading
    make run-questions ARGS="--ids nlp07,nlp27"          # a few questions
    make run-questions ARGS="--path baseline"            # Fast (baseline-enhanced = Adaptive)
    make run-questions ARGS="--parallel 3"               # questions at a time (default 2)
    make run-questions ARGS="--resume results/run_questions-<time>.jsonl"   # skip what already answered

Needs the API (`make demo` or `make dev`) and the models. For Agentic, each question first gets a
proposal run (the readings you would see in the UI); then one research run is started per reading,
exactly as picking it in the UI would. Results are appended to results/run_questions-<time>.jsonl as
they finish, so an interrupted batch keeps its progress. Afterwards: make graph-shape ARGS="-n 200".
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

ROOT = Path(__file__).resolve().parents[1]
FINISHED = {"answered", "corpus_absent", "no_evidence_found", "awaiting_reading", "incomplete", "stopped"}
write_lock = threading.Lock()


def wait(client: httpx.Client, run_id: str, timeout: float) -> dict[str, Any]:
    """Poll until the run is written (404 while it runs) and has a final status."""
    started = time.monotonic()
    while True:
        response = client.get(f"/runs/{run_id}")
        if response.status_code == 200 and response.json().get("status") in FINISHED:
            return response.json()
        if response.status_code not in {200, 404}:
            response.raise_for_status()
        if time.monotonic() - started > timeout:
            raise TimeoutError(f"{run_id} did not finish within {timeout:.0f}s")
        time.sleep(2)


def start(client: httpx.Client, body: dict[str, Any]) -> str:
    response = client.post("/runs", json=body)
    response.raise_for_status()
    return str(response.json()["id"])


def summary(run: dict[str, Any]) -> dict[str, Any]:
    chosen = run.get("chosenReading")
    readings = run.get("readings") or []
    reading = readings[chosen] if isinstance(chosen, int) and 0 <= chosen < len(readings) else {}
    graph = reading.get("graph") or {}
    return {"status": run.get("status"), "reason": run.get("reason"), "duration_s": run.get("durationS"),
            "facts": len(reading.get("triples") or []), "graph_edges": len(graph.get("edges") or []),
            "execution_status": run.get("executionStatus")}


def main(
    questions: Annotated[Path, typer.Option(help="Question set (JSON list with id and question).")] =
        ROOT / "datasets" / "question_library_2025.json",
    ids: Annotated[str | None, typer.Option(help="Only these question ids, comma-separated.")] = None,
    path: Annotated[str, typer.Option(help="agentic, baseline or baseline-enhanced.")] = "agentic",
    budget: Annotated[str, typer.Option(help="fast or slow.")] = "slow",
    parallel: Annotated[int, typer.Option(help="Questions processed at the same time.")] = 2,
    api: Annotated[str, typer.Option(help="Harness API base URL.")] = "http://127.0.0.1:8000",
    timeout: Annotated[float, typer.Option(help="Seconds to wait for one run.")] = 3600,
    resume: Annotated[Path | None, typer.Option(help="Earlier results file: skip answered question/reading pairs.")] = None,
) -> None:
    rows = json.loads(questions.read_text(encoding="utf-8"))
    wanted = {part.strip() for part in ids.split(",")} if ids else None
    rows = [row for row in rows if wanted is None or row["id"] in wanted]
    done: set[tuple[str, str]] = set()
    if resume and resume.exists():
        for line in resume.read_text().splitlines():
            item = json.loads(line)
            if item.get("status") == "answered":
                done.add((item["question_id"], item.get("reading") or ""))
    out = resume or ROOT / "results" / f"run_questions-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    typer.echo(f"{len(rows)} questions · path {path} · budget {budget} · results in {out}")

    def record(item: dict[str, Any]) -> None:
        with write_lock, out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        typer.echo(f"[{item['question_id']}] {item.get('reading') or '-'} → {item.get('run_id') or '-'} "
                   f"{item.get('status')} {item.get('duration_s') or ''}s {item.get('facts', '')} facts"
                   + (f" ({item['reason']})" if item.get("reason") else ""))

    def one(row: dict[str, Any]) -> None:
        base = {"question_id": row["id"], "question": row["question"], "path": path, "budget": budget}
        with httpx.Client(base_url=api, timeout=60) as client:
            try:
                first = start(client, {"question": row["question"], "path": path, "budget": budget})
                run = wait(client, first, timeout)
            except (httpx.HTTPError, TimeoutError) as exc:
                record({**base, "reading": None, "run_id": None, "status": "error", "reason": str(exc)[:300]})
                return
            concepts = run.get("concepts") or []
            if path != "agentic" or run.get("status") != "awaiting_reading" or not concepts:
                # Fast/Adaptive, or an agentic question answered without readings (verified facts).
                record({**base, "reading": run.get("concept", {}) and run["concept"].get("label"), "run_id": first,
                        **summary(run)})
                return
            for index, concept in enumerate(concepts):
                label = str(concept.get("label") or f"reading {index + 1}")
                if (row["id"], label) in done:
                    continue
                try:
                    child = start(client, {"question": row["question"], "path": path, "budget": budget,
                                           "concept": concept, "parent_run_id": first})
                    result = wait(client, child, timeout)
                    record({**base, "reading": label, "reading_index": index, "readings": len(concepts),
                            "proposal_id": first, "run_id": child, **summary(result)})
                except (httpx.HTTPError, TimeoutError) as exc:
                    record({**base, "reading": label, "reading_index": index, "proposal_id": first,
                            "run_id": None, "status": "error", "reason": str(exc)[:300]})

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
        list(pool.map(one, rows))
    results = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    statuses: dict[str, int] = {}
    for item in results:
        statuses[str(item.get("status"))] = statuses.get(str(item.get("status")), 0) + 1
    typer.echo(f"done in {(time.monotonic() - started) / 60:.1f} min: "
               + ", ".join(f"{count} {status}" for status, count in sorted(statuses.items())))
    typer.echo('Next: make graph-shape ARGS="-n 200 --path agentic"')


if __name__ == "__main__":
    typer.run(main)
