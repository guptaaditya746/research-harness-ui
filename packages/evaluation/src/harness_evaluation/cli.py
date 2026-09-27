"""Offline evaluator for completed artifact runs."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from harness_schema.models import Run

from . import evaluate_run


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _gold_by_question(dataset: Path) -> dict[str, list[dict]]:
    return {
        row["question"]: row.get("gold_triples", [])
        for row in _rows(dataset)
        if row.get("question")
    }


def evaluate_directory(artifacts: Path, dataset: Path, path: str | None = None) -> list[dict]:
    """Evaluate runs whose question occurs in the JSONL gold dataset."""

    gold = _gold_by_question(dataset)
    results: list[dict] = []
    for run_file in sorted(artifacts.glob("*/run.json")):
        run = Run.model_validate_json(run_file.read_text())
        if path and run.path != path:
            continue
        if run.question not in gold:
            continue
        score = evaluate_run(run, gold[run.question])
        chosen = score["chosen"] or {}
        results.append(
            {
                "run_id": run.id,
                "question": run.question,
                "path": run.path,
                "budget": run.budget,
                "status": run.status,
                "precision": chosen.get("precision", 0.0),
                "recall": chosen.get("recall", 0.0),
                "f1": chosen.get("f1", 0.0),
                "provenance_coverage": chosen.get("provenance_coverage", 0.0),
                "unsupported_ratio": chosen.get("unsupported_ratio", 0.0),
                "answer_supported": chosen.get("answer_supported", False),
            }
        )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="harness-eval")
    parser.add_argument("run", nargs="?", default="run")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--path", choices=["agentic", "baseline", "baseline-enhanced", "external"])
    parser.add_argument("--repetitions", type=int, default=1, help="expected repetitions in evaluation mode")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.run != "run":
        parser.error("the only supported command is: harness-eval run")
    results = evaluate_directory(args.artifacts, args.dataset, args.path)
    payload = {
        "computed_at": datetime.now(UTC).isoformat(),
        "results": results,
    }
    output = args.output or Path("results") / datetime.now(UTC).strftime("%Y%m%d-%H%M%S") / "scores.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {len(results)} scores to {output}")
    for row in results:
        print(f"{row['path']:18} {row['run_id']}  P={row['precision']:.2f} R={row['recall']:.2f} F1={row['f1']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
