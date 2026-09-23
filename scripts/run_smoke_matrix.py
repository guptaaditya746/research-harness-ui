"""Repeat a question matrix against a running harness API, saving every result."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx


def run_one(
    client: httpx.Client, row: dict, path: str, repetition: int, sequence: int, timeout: float
) -> dict:
    started = time.monotonic()
    run_id = None
    shared = {
        "sequence": sequence,
        "question_id": row["id"],
        "type": row["type"],
        "category": row["category"],
        "expected": row["expected"],
        "question": row["question"],
        "path": path,
        "repetition": repetition,
    }
    try:
        response = client.post(
            "/runs", json={"question": row["question"], "path": path, "budget": "slow"}
        )
        response.raise_for_status()
        run_id = response.json()["id"]
        while True:
            response = client.get(f"/runs/{run_id}")
            if response.status_code == 200:
                run = response.json()
                break
            if response.status_code != 404:
                response.raise_for_status()
            if time.monotonic() - started >= timeout:
                raise TimeoutError(f"run {run_id} exceeded {timeout}s")
            time.sleep(0.5)
        chosen = run.get("chosenReading")
        return {
            **shared,
            "id": run_id,
            "status": run["status"],
            "reason": run.get("reason"),
            "duration_s": run["durationS"],
            "elapsed_s": round(time.monotonic() - started, 3),
            "triple_count": sum(len(reading["triples"]) for reading in run["readings"]),
            "answer": run["readings"][chosen]["answer"] if chosen is not None else None,
            "reading_labels": [reading["label"] for reading in run["readings"]],
        }
    except (httpx.HTTPError, TimeoutError, KeyError, ValueError) as error:
        return {
            **shared,
            "id": run_id,
            "status": "harness_error",
            "reason": str(error),
            "elapsed_s": round(time.monotonic() - started, 3),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--manifest", type=Path, default=Path("datasets/question_matrix_20.json"))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--output", type=Path, default=Path("docs/evidence/question-matrix-20.json"))
    args = parser.parse_args()
    if args.repetitions < 1 or args.concurrency < 1:
        parser.error("repetitions and concurrency must be positive")
    questions = json.loads(args.manifest.read_text())
    if len({row["type"] for row in questions}) != len(questions):
        parser.error("question types must be unique")
    paths = ("baseline", "baseline-enhanced", "agentic")
    work = [
        (row, path, repetition, sequence)
        for sequence, (row, path, repetition) in enumerate(
            (row, path, repetition)
            for row in questions
            for path in paths
            for repetition in range(1, args.repetitions + 1)
        )
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict] = json.loads(args.output.read_text()) if args.output.exists() else []
    sequence_by_key = {
        (row["id"], path, repetition): sequence
        for row, path, repetition, sequence in work
    }
    for result in results:
        key = (result["question_id"], result["path"], result["repetition"])
        result["sequence"] = sequence_by_key.get(key, result["sequence"])
    completed = {
        (row["question_id"], row["path"], row["repetition"])
        for row in results
    }
    pending = [
        item for item in work
        if (item[0]["id"], item[1], item[2]) not in completed
    ]

    with (
        httpx.Client(base_url=args.base_url, timeout=20) as client,
        ThreadPoolExecutor(max_workers=args.concurrency) as executor,
    ):
        futures = {
            executor.submit(run_one, client, row, path, repetition, sequence, args.timeout): sequence
            for row, path, repetition, sequence in pending
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            results.sort(key=lambda row: row["sequence"])
            args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
            print(
                f"{len(results)}/{len(work)} {result['question_id']} "
                f"{result['path']} #{result['repetition']}: {result['status']} "
                f"{result.get('reason') or ''} ({result['elapsed_s']:.1f}s)",
                flush=True,
            )


if __name__ == "__main__":
    main()
