"""How the answer graphs of recent runs are shaped (read-only; Phase 4 diagnostics).

    make graph-shape                              # latest 30 runs, a table per run and medians per path
    make graph-shape ARGS="-n 100 --path agentic"
    make graph-shape ARGS="--run r-1a2b3c4d"      # one run, with examples of dead ends
    make graph-shape ARGS="--json --out results/graph_shape.json"

Reads run.json, subschema.ttl and triples.ttl under $ARTIFACT_ROOT; writes nothing unless --out is
given. What each number means: packages/evaluation/src/harness_evaluation/graph_shape.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from harness_config import settings
from harness_evaluation.graph_shape import shape, summarize
from harness_runs.analysis import load_runs
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
console = Console()
PATHS = {"baseline": "Fast", "baseline-enhanced": "Adapt", "agentic": "Agent"}
LEGEND = ("Facts = fact edges in the answer graph = BB (DBLP backbone, added) + Ext (extracted by the run); DBLP = share of "
          "all facts using DBLP predicates; Inv = share of extracted facts using invented predicates; Pieces = "
          "disconnected parts; Pairs = paper pairs connected at all (– when fewer than 2 papers); Dead = invented "
          "edges whose object nothing else touches; Stmt = extracted objects that are whole statements (>8 words); "
          "Funnel = passages read > open propositions > mapped onto the schema > kept by evidence review; "
          "Export = triples in triples.ttl.")


def _pct(value: float | None) -> str:
    return "–" if value is None else f"{value * 100:.0f}%"


def _num(value: float | None) -> str:
    return "–" if value is None else f"{value:g}"


def main(
    n: Annotated[int, typer.Option("-n", help="How many recent runs.")] = 30,
    path: Annotated[str | None, typer.Option(help="baseline, baseline-enhanced or agentic.")] = None,
    run: Annotated[str | None, typer.Option("--run", help="One run id (shows examples).")] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print everything as JSON.")] = False,
    out: Annotated[Path | None, typer.Option(help="Also write the JSON to this file.")] = None,
) -> None:
    root = Path(settings().artifact_root or ROOT / "artifacts")
    runs = [item for item in load_runs(root) if item.status == "answered"]
    if run:
        runs = [item for item in runs if item.id == run]
    if path:
        runs = [item for item in runs if item.path == path]
    rows = [row for row in (shape(item, root) for item in runs[: max(1, n)]) if row is not None]
    result = {"artifact_root": str(root), "runs": rows, "summary": summarize(rows)}
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    if as_json:
        console.print_json(json.dumps(result, ensure_ascii=False))
        return
    if not rows:
        console.print(f"No answered runs with a reading under {root}.")
        return

    table = Table(title=f"Answer graphs of {len(rows)} answered runs (chosen reading)")
    for column in ("Run", "Kind", "Path", "Facts", "BB", "Ext", "DBLP", "Inv", "Pieces", "Pairs", "Dead", "Stmt",
                   "Funnel P>O>M>K", "Export"):
        table.add_column(column, justify="left" if column in {"Run", "Kind", "Path"} else "right",
                         no_wrap=column == "Run")
    for row in rows:
        export = row.get("export") or {}
        flow = row.get("funnel")
        table.add_row(row["id"], row["kind"], PATHS.get(row["path"], row["path"]), str(row["fact_edges"]),
                      str(row["backbone_edges"]), str(row["extracted_edges"]),
                      _pct(row["dblp_share"]), _pct(row["extracted_invented_share"]), str(row["pieces"]),
                      _pct(row["paper_pairs_connected"]), _pct(row["invented_dead_end_share"]),
                      _pct(row["statement_object_share"]),
                      f"{flow['passages']}>{flow['open']}>{flow['mapped']}>{flow['kept']}" if flow else "–",
                      str(export.get("triples", "–")))
    console.print(table)

    summary = Table(title="Medians per kind and path (research = answered from passages, facts = from DBLP)")
    for column in ("Kind", "Path", "Runs", "Facts", "BB", "Ext", "DBLP", "Inv", "Pieces", "Pairs", "Dead", "Stmt",
                   "Funnel P>O>M>K", "New classes"):
        summary.add_column(column, justify="left" if column in {"Kind", "Path"} else "right")
    for item in result["summary"]:
        flow = item.get("funnel")
        summary.add_row(item["kind"], PATHS.get(item["path"], item["path"]), str(item["runs"]),
                        _num(item["fact_edges"]), _num(item["backbone_edges"]), _num(item["extracted_edges"]),
                        _pct(item["dblp_share"]), _pct(item["invented_share"]), _num(item["pieces"]),
                        _pct(item["paper_pairs_connected"]), _pct(item["invented_dead_end_share"]),
                        _pct(item["statement_object_share"]),
                        f"{_num(flow['passages'])}>{_num(flow['open'])}>{_num(flow['mapped'])}>{_num(flow['kept'])}"
                        if flow else "–", str(item["runs_with_invented_classes"]))
    console.print(summary)
    console.print(f"[dim]{LEGEND} New classes = runs that invented at least one class.[/dim]")
    everything = next((item for item in result["summary"] if item["kind"] == "research" and item["path"] == "all"),
                      result["summary"][-1])
    if everything["top_invented_properties"]:
        console.print("Most invented properties in the runs' schemas: " + ", ".join(
            f"{name} ({count})" for name, count in everything["top_invented_properties"]))

    schemas = [row["schema"] for row in rows if row.get("schema") and "error" not in row["schema"]]
    if schemas:
        invented = sum(len(item["invented_properties"]) for item in schemas)
        console.print(
            f"Invented properties in subschema.ttl: {invented}; typed as OWL 2 object/datatype properties: "
            f"{sum(item['owl_typed'] for item in schemas)}; with a parent property: "
            f"{sum(item['with_parent'] for item in schemas)}; with domain and range: "
            f"{sum(item['with_domain_and_range'] for item in schemas)}. Invented classes: "
            f"{sum(len(item['invented_classes']) for item in schemas)}.")
    if run and rows:
        row = rows[0]
        console.print(f"\n[bold]{row['question']}[/bold]  (reading: {row['reading']})")
        for kind, lines in row["examples"].items():
            for line in lines:
                console.print(f"  {kind}: {line}")
        console.print(f"  invented properties: {', '.join(row['invented_properties']) or 'none'}")
        console.print(f"  invented classes: {', '.join(row['invented_classes']) or 'none'}")


if __name__ == "__main__":
    typer.run(main)
