"""List few-shot example candidates from errors on the development part of the gold set.

    make fewshot ARGS="--dev nlp01,nlp04,nlp09"      # development question ids
    make fewshot ARGS="--dev @questions/dev_ids.txt"  # one id per line
    make fewshot ARGS="--dev nlp01 --include-okay --out results/fewshot.json"

Missed vital gold triples become `extraction/edc_extract` candidates; extra triples that reviewers
rejected as wrong become `extraction/evidence_review` candidates. Repeated errors come first. Pick a
few, write their "why", and copy {input, output, why} into
packages/prompts/src/harness_prompts/examples/v2/<prompt>.json. Soft matches use the judge cache
only (no model calls). Never take examples from questions you report results on (docs/PROMPTS.md).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from harness_config import settings
from harness_evaluation import load_relations
from harness_evaluation.fewshot import fewshot_candidates
from harness_evaluation.judge import CachedJudge
from harness_runs.analysis import load_runs
from harness_runs.store import GoldCandidateStore
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
console = Console()


def _ids(value: str | None) -> set[str] | None:
    if not value:
        return None
    if value.startswith("@"):
        lines = Path(value[1:]).read_text(encoding="utf-8").splitlines()
        return {line.split("#")[0].strip() for line in lines if line.split("#")[0].strip()}
    return {part.strip() for part in value.split(",") if part.strip()}


def _library(datasets: Path) -> dict[str, str]:
    """Question ids -> texts from the harness question libraries, to name older gold rows."""
    texts: dict[str, str] = {}
    for path in sorted(datasets.glob("question_*.json")):
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for row in rows if isinstance(rows, list) else []:
            if isinstance(row, dict) and row.get("id") and row.get("question"):
                texts.setdefault(str(row["id"]), str(row["question"]))
    return texts


def main(
    dev: Annotated[str | None, typer.Option(help="Development question ids: a,b,c or @file.")] = None,
    include_okay: Annotated[bool, typer.Option("--include-okay", help="Also list missed non-vital triples.")] = False,
    top: Annotated[int, typer.Option(help="Rows shown per prompt.")] = 15,
    out: Annotated[Path, typer.Option(help="Full list as JSON.")] = ROOT / "results" / "fewshot_candidates.json",
) -> None:
    datasets = Path(settings().datasets_root or ROOT / "datasets")
    candidates = [row for row in GoldCandidateStore(datasets / "gold" / "candidates.jsonl").list()
                  if row.get("status") == "approved"]
    question_ids = _ids(dev)
    if question_ids is None:
        console.print("[yellow]No --dev ids: listing errors on every approved question. Take examples only "
                      "from questions you will not report results on.[/yellow]")
    runs = load_runs(settings().artifact_root or ROOT / "artifacts")
    result = fewshot_candidates(candidates, runs, relations=load_relations(), judge=CachedJudge(),
                                question_ids=question_ids, include_okay=include_okay,
                                question_texts=_library(datasets))
    if result["unmatched_ids"]:
        console.print(f"[yellow]No approved gold row for: {', '.join(result['unmatched_ids'])} "
                      "(ids are the rows' question_id, the question library id or the row id).[/yellow]")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    console.print(f"{len(result['questions'])} questions, {result['scored_runs']} scored runs; full list: {out}")
    for prompt, rows in result["candidates"].items():
        table = Table(title=f"{prompt}: {len(rows)} candidates")
        table.add_column("Errors", justify="right")
        table.add_column("Question")
        table.add_column("Triple")
        table.add_column("Source")
        table.add_column("Paths")
        for row in rows[:top]:
            triple = row["triple"]
            table.add_row(str(row["errors"]), row["question_id"], f"({triple['s']}, {triple['p']}, {triple['o']})"[:90],
                          str(row.get("source") or "–"), ", ".join(f"{k} {v}" for k, v in row["by_path"].items()))
        console.print(table)


if __name__ == "__main__":
    typer.run(main)
