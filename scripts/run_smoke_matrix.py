"""Repeat the recorded smoke questions against a running harness API."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--output", type=Path, default=Path("docs/evidence/smoke-matrix-rerun.json"))
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("repetitions must be positive")
    previous = json.loads(Path("docs/evidence/smoke-matrix.json").read_text())
    questions = list(dict.fromkeys(row["question"] for row in previous))
    paths = ("baseline", "baseline-enhanced", "agentic")
    results: list[dict] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with httpx.Client(base_url=args.base_url, timeout=20) as client:
        for question in questions:
            for path in paths:
                for repetition in range(1, args.repetitions + 1):
                    started = time.monotonic()
                    run_id = None
                    try:
                        response = client.post(
                            "/runs",
                            json={"question": question, "path": path, "budget": "slow"},
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
                            if time.monotonic() - started >= args.timeout:
                                raise TimeoutError(f"run {run_id} exceeded {args.timeout}s")
                            time.sleep(0.5)
                        result = {
                            "question": question,
                            "path": path,
                            "repetition": repetition,
                            "id": run_id,
                            "status": run["status"],
                            "reason": run.get("reason"),
                            "duration_s": run["durationS"],
                            "elapsed_s": round(time.monotonic() - started, 3),
                            "triple_count": sum(len(reading["triples"]) for reading in run["readings"]),
                            "answer": run["readings"][run["chosenReading"]]["answer"]
                            if run.get("chosenReading") is not None else None,
                        }
                    except (httpx.HTTPError, TimeoutError, KeyError, ValueError) as error:
                        result = {
                            "question": question,
                            "path": path,
                            "repetition": repetition,
                            "id": run_id,
                            "status": "harness_error",
                            "reason": str(error),
                            "elapsed_s": round(time.monotonic() - started, 3),
                        }
                    results.append(result)
                    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
                    print(
                        f"{len(results)}/{len(questions) * len(paths) * args.repetitions} "
                        f"{path} #{repetition}: {result['status']} "
                        f"{result.get('reason') or ''} ({result['elapsed_s']:.1f}s)",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
