"""Measure query-planner accuracy against the live model and corpus.

Usage (on the compute node, with .env loaded and corpus_data present):
    uv run python scripts/eval_planner.py [--cases datasets/planner_cases.json]

For every case it checks the chosen operation, that each expected paper is
resolved from the plan's mentions, and (when given) the evidence focus. It
prints one line per case and exits non-zero below --min-accuracy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from harness_baseline.planner import PaperResolver, QueryPlanner
from harness_corpus.store import CorpusStore
from harness_extraction import OpenAICompatibleModel

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=str(ROOT / "datasets" / "planner_cases.json"))
    parser.add_argument("--corpus", default=os.getenv("CORPUS_ROOT", str(ROOT / "corpus_data")))
    parser.add_argument("--min-accuracy", type=float, default=0.85)
    args = parser.parse_args()

    corpus = CorpusStore(Path(args.corpus))
    titles = {str(r["subject"]): str(r["object"]) for r in corpus.seed_facts() if r.get("predicate") == "title"}
    titles.update({pid: str(p.get("title") or "") for pid, p in corpus.by_paper.items() if pid not in titles})
    resolver = PaperResolver(titles)
    model = OpenAICompatibleModel(timeout_s=float(os.getenv("BUDGET_BASELINE_SLOW_SECONDS", "90")))
    planner = QueryPlanner(model)
    print(f"model={model.model} base={model.base_url} papers={len(titles)}")

    cases = json.loads(Path(args.cases).read_text())
    passed = 0
    for case in cases:
        started = time.monotonic()
        plan = planner.plan(case["question"])
        resolved = [resolver.resolve(m) for m in plan.papers]
        problems = []
        if plan.operation not in case["operation"]:
            problems.append(f"op={plan.operation} want {'|'.join(case['operation'])}")
        missing = [p for p in case.get("papers", []) if p not in resolved]
        if missing and (plan.operation != "research" or case.get("papers")):
            problems.append(f"papers={plan.papers}->{resolved} want {case['papers']}")
        if case.get("evidence_focus") and plan.evidence_focus != case["evidence_focus"]:
            problems.append(f"focus={plan.evidence_focus} want {case['evidence_focus']}")
        if plan.source == "fallback":
            problems.append(f"fallback: {plan.rationale}")
        ok = not problems
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'} {time.monotonic() - started:5.1f}s  {case['question'][:70]}"
              + ("" if ok else "\n      " + "; ".join(problems)))
    accuracy = passed / len(cases) if cases else 0.0
    print(f"\n{passed}/{len(cases)} correct ({accuracy:.0%})")
    return 0 if accuracy >= args.min_accuracy else 1


if __name__ == "__main__":
    sys.exit(main())
