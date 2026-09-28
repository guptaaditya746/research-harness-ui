"""Score saved runs against the approved gold: strict / entailed / soft, by complexity and path.

    make evaluate                          # report from the judge cache only (no model calls)
    make evaluate ARGS="--judge"           # also ask the soft-match judge (HARNESS_MODEL_JUDGE) and cache it
    make evaluate ARGS="--variants-only"   # soft level without the judge (judge failed its agreement check)
    make evaluate ARGS="--json --out results/benchmark.json"

Reads the gold from $DATASETS_ROOT/gold/candidates.jsonl (installed by `labelling install`) and the
runs from $ARTIFACT_ROOT. The same numbers appear on the Benchmark page. Metric definitions,
reasons and limits: labelling/docs/EVALUATION.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from harness_config import settings
from harness_evaluation import LEVELS, load_relations
from harness_evaluation.benchmark import build_benchmark
from harness_evaluation.judge import CachedJudge
from harness_runs.analysis import load_runs
from harness_runs.store import GoldCandidateStore
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
console = Console()
PATH_LABELS = {"baseline": "Fast", "adaptive-baseline": "Adaptive", "agentic": "Agentic", "external": "External"}


def _interval(value: dict | None) -> str:
    if not value or value.get("mean") is None:
        return "–"
    if value["low"] == value["high"]:
        return f"{value['mean']:.2f}"
    return f"{value['mean']:.2f} [{value['low']:.2f}–{value['high']:.2f}]"


def _judge(live: bool, off: bool) -> CachedJudge | None:
    if off:
        return None
    if not live:
        return CachedJudge()
    from harness_extraction import OpenAICompatibleModel, complete

    model = OpenAICompatibleModel(model=settings().model_judge, timeout_s=120)
    return CachedJudge(complete=lambda prompt: complete(model, prompt))


def main(
    judge: Annotated[bool, typer.Option("--judge", help="Ask the soft-match judge for unseen pairs.")] = False,
    variants_only: Annotated[bool, typer.Option("--variants-only", help="Soft level without the judge.")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print the full result as JSON.")] = False,
    out: Annotated[Path | None, typer.Option(help="Also write the full result to this file.")] = None,
    gold_version: Annotated[str | None, typer.Option(help="Only gold rows of this version (e.g. pilot-v1).")] = None,
) -> None:
    datasets = settings().datasets_root or ROOT / "datasets"
    candidates = [row for row in GoldCandidateStore(Path(datasets) / "gold" / "candidates.jsonl").list()
                  if row.get("status") == "approved" and (gold_version is None or row.get("gold_version") == gold_version)]
    runs = load_runs(settings().artifact_root or ROOT / "artifacts")
    if judge and variants_only:
        raise typer.BadParameter("--judge and --variants-only exclude each other")
    scorer = _judge(judge, variants_only)
    result = build_benchmark(candidates, runs, relations=load_relations(), judge=scorer)
    result["judge"] = "off" if scorer is None else ("live" if judge else "cache")
    result["judge_calls"] = scorer.calls if scorer else 0
    # Who labelled the gold: people (Argilla), computed (structural) or a model (silver versions).
    sources = sorted({str(row.get("label_source") or "people") for row in candidates})
    result["gold_label_sources"] = sources
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, default=str) + "\n")
    if as_json:
        console.print_json(json.dumps(result, default=str))
        return
    console.print(f"{len(candidates)} approved gold questions, {len(runs)} runs; rules: "
                  f"{', '.join(result['relations'])}")
    if "model" in sources:
        console.print("[bold yellow]Silver gold: some rows were labelled by a model, not by people "
                      "(label_source: model). These numbers are provisional.[/]")
    table = Table(title="Triple F1 by complexity and path (mean [95% CI] over questions)")
    table.add_column("Complexity")
    table.add_column("Level")
    for label in PATH_LABELS.values():
        table.add_column(label)
    for row in result["by_complexity"]["rows"]:
        for index, level in enumerate((*LEVELS, "vital_recall")):
            first = f"{row['complexity']} ({row['questions']})" if index == 0 else ""
            table.add_row(first, level, *(_interval(row["paths"][path][level]) for path in PATH_LABELS))
        table.add_section()
    console.print(table)
    differences = Table(title="Soft F1 differences, paired over questions (mean [95% bootstrap CI])")
    differences.add_column("Complexity")
    everything = result["by_complexity"]["rows"]
    names = list(everything[0]["soft_f1_differences"]) if everything else []
    for name in names:
        differences.add_column(name)
    for row in everything:
        differences.add_row(f"{row['complexity']} ({row['questions']})",
                            *(_interval(row["soft_f1_differences"][name]) for name in names))
    console.print(differences)
    pending = sum(entry["judge_pending"] for row in result["questions"] for entry in row["paths"])
    if pending:
        console.print(f"[yellow]{pending} judge pairs are not in the cache; soft scores are lower bounds "
                      f"until `make evaluate ARGS=--judge` runs.[/yellow]")
    per_question = Table(title="Soft F1 per question (mean over repetitions)")
    per_question.add_column("Question")
    per_question.add_column("Complexity")
    for label in PATH_LABELS.values():
        per_question.add_column(label)
    for row in result["questions"]:
        cells = []
        for path in PATH_LABELS:
            entry = next((item for item in row["paths"] if item["path"] == path), None)
            value = entry["levels"]["soft"]["f1"]["mean"] if entry else None
            cells.append("–" if value is None else f"{value:.2f}")
        per_question.add_row(str(row["question"])[:70], str(row.get("complexity") or "–"), *cells)
    console.print(per_question)


if __name__ == "__main__":
    typer.run(main)
