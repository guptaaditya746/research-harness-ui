"""Why a gold row scores what it scores: pairing, statuses, and what its triples have in common with the runs'.

``make evaluate ARGS=--explain`` prints this per gold row. Each step can turn a score into zero or "–":

1. pairing: a run counts for a row only with the same question text, path and evaluation snapshot;
2. status: only answered, non-degraded runs are scored;
3. triples: strict needs equal canonical keys, entailed a rule, soft a rule, a variant or the judge; the
   judge is only asked about pairs with the same subject, and only from its cache unless ``--judge``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from difflib import get_close_matches
from typing import Any

from harness_schema.models import Run

from . import canonical_triple_key
from .matching import Judge, _ask_judge, _pairs
from .relations import PredicateRelations

PATHS = ("baseline", "baseline-enhanced", "agentic", "external")
LOCAL_PAPER = "https://harness.local/resource/p"
DBLP_REC = "https://dblp.org/rec/"


def _key(item: Mapping[str, Any]) -> tuple[str, str, str]:
    return canonical_triple_key(item)


def _gold_sets(row: Mapping[str, Any]) -> list[tuple[str, list[Mapping[str, Any]]]]:
    sets = [("row", list(row.get("gold_triples") or []))]
    sets += [(f"reading {item.get('reading_id') or item.get('label')}", list(item.get("gold_triples") or []))
             for item in row.get("accepted_readings") or []]
    return [(name, triples) for name, triples in sets if triples]


def _top(values: Iterable[str], n: int = 5) -> list[list[Any]]:
    return [[value, count] for value, count in Counter(values).most_common(n)]


def _snapshot_diff(row: Mapping[str, Any], run: Run) -> dict[str, list[Any]]:
    wanted = row.get("snapshot") or {}
    have = run.snapshot.model_dump(by_alias=True) if run.snapshot else {}
    return {name: [wanted.get(name), have.get(name)] for name in sorted(set(wanted) | set(have))
            if wanted.get(name) != have.get(name)}


def _compare(gold: list[Mapping[str, Any]], run: Run, relations: PredicateRelations,
             judge: Judge | None) -> dict[str, Any]:
    reading = run.readings[run.chosen_reading] if run.readings and run.chosen_reading is not None \
        and run.chosen_reading < len(run.readings) else None
    triples = [item.model_dump() for item in reading.triples] if reading else []
    pairs = _pairs(triples, gold, _key, relations)
    _, asked, pending = _ask_judge(pairs, judge)
    gold_subjects = {key[0] for key in pairs.gold_keys}
    run_subjects = {key[0] for key in pairs.run_keys}
    return {
        "run": run.id,
        "run_triples": len(pairs.run_keys),
        "excluded_dblp": len(pairs.excluded),
        "run_subjects": _top(key[0] for key in pairs.run_keys),
        "run_predicates": _top(key[1] for key in pairs.run_keys),
        "shared_subjects": len(gold_subjects & run_subjects),
        "shared_predicates": len({key[1] for key in pairs.gold_keys} & {key[1] for key in pairs.run_keys}),
        "strict_pairs": len(pairs.strict),
        "rule_pairs": len(pairs.entailed_precision | pairs.entailed_recall),
        "variant_pairs": len(pairs.variants),
        "judge_pairs": asked,
        "judge_pending": pending,
    }


def explain(row: Mapping[str, Any], runs: Iterable[Run], *, relations: PredicateRelations,
            judge: Judge | None = None, run_input: str = "corpus") -> dict[str, Any]:
    """Diagnostics for one gold row (see the module docstring), with plain-language ``hints``. Only runs of
    ``run_input`` are paired; runs of the other input are counted in a hint."""
    every = list(runs)
    runs = [run for run in every if run.run_input == run_input]
    other = sum(1 for run in every if run.run_input != run_input and run.question == row.get("question"))
    question = row.get("question")
    same = [run for run in runs if run.question == question]
    hints: list[str] = []
    if other:
        hints.append(f"{other} runs of this question read the other input (not {run_input}); "
                     f"they are scored with --input {'corpus' if run_input != 'corpus' else 'evidence_doc'}")
    if not same:
        near = get_close_matches(str(question), sorted({run.question for run in runs}), n=3, cutoff=0.6)
        hints.append("no saved run has this exact question text"
                     + (f"; closest: {near}" if near else "; run `make benchmark-runs` for it"))
    gold_sets = _gold_sets(row)
    gold = [item for _, items in gold_sets for item in items]
    gold_keys = [_key(item) for item in gold]
    paths: dict[str, Any] = {}
    for path in PATHS:
        mine = [run for run in same if run.path == path]
        if not mine:
            continue
        paired = [run for run in mine if row.get("snapshot") is None
                  or (run.snapshot and run.snapshot.model_dump(by_alias=True) == row["snapshot"])]
        statuses = Counter(run.status for run in paired)
        entry: dict[str, Any] = {"runs": len(mine), "paired": len(paired), "statuses": dict(statuses)}
        unstamped = [run for run in mine if run.snapshot is None and run not in paired]
        if unstamped:
            hints.append(f"{path}: {len(unstamped)} runs have no evaluation snapshot (started outside "
                         "`make benchmark-runs` / Benchmark → Run), so they are never scored")
        if len(paired) + len(unstamped) < len(mine):
            entry["snapshot_differs"] = _snapshot_diff(
                row, next(run for run in mine if run not in paired and run not in unstamped))
            hints.append(f"{path}: {len(mine) - len(paired) - len(unstamped)} of {len(mine)} runs have another "
                         "evaluation snapshot "
                         f"(gold, run): {entry['snapshot_differs']}; set the same CORPUS_SNAPSHOT_ID, "
                         "ONTOLOGY_VERSION and RETRIEVAL_INDEX_VERSION in labelling and harness .env, then "
                         "re-export the gold or rerun")
        answered = [run for run in paired if run.status == "answered"]
        if paired and not answered:
            hints.append(f"{path}: no paired run answered ({dict(statuses)}); only answered runs are scored")
        if answered and gold:
            entry["first_answered"] = comparison = _compare(gold, answered[0], relations, judge)
            if comparison["run_triples"] == 0:
                hints.append(f"{path}: run {comparison['run']} has no triples in its chosen reading")
            elif comparison["shared_subjects"] == 0:
                hints.append(f"{path}: gold and run {comparison['run']} share no subject, so no pair can match and "
                             "the judge is never asked (gold subjects "
                             f"{[s for s, _ in _top((k[0] for k in gold_keys), 3)]}, run subjects "
                             f"{[s for s, _ in comparison['run_subjects'][:3]]})")
                gold_papers = sum(k[0].startswith(LOCAL_PAPER) for k in gold_keys)
                run_dblp = sum(subject.startswith(DBLP_REC) for subject, _ in comparison["run_subjects"])
                if gold_papers and run_dblp:
                    hints.append("the gold names papers by their local key (p…) where the run uses their DBLP "
                                 "IRI: the gold was exported without canonical paper IRIs; update labelling "
                                 "and export (or draft) again")
            elif comparison["judge_pending"]:
                hints.append(f"{path}: {comparison['judge_pending']} same-subject pairs await the judge; "
                             "`make evaluate ARGS=--judge` asks it (soft scores are lower bounds until then)")
            if comparison["shared_predicates"] == 0 and comparison["run_triples"]:
                hints.append(f"{path}: no predicate in common with the gold, so strict is 0 by construction; "
                             "only rules and the judge (soft) can match")
        paths[path] = entry
    if not gold:
        hints.append("the gold row has no gold triples")
    return {
        "id": row.get("id"),
        "question": question,
        "gold_version": row.get("gold_version"),
        "gold_triples": {name: len(items) for name, items in gold_sets},
        "gold_subjects": _top(key[0] for key in gold_keys),
        "gold_predicates": _top(key[1] for key in gold_keys),
        "paths": paths,
        "hints": hints,
    }


__all__ = ["explain"]
