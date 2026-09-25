"""Few-shot candidates from gold failures (Phase 3f).

Two kinds of error on the gold set become example candidates:

- ``extraction/edc_extract``: vital gold triples (optionally all gold triples) that runs missed
  even at the soft level. The draft shows the gold triple's evidence passage as input and the
  proposition that should have been extracted from it as output.
- ``extraction/evidence_review``: extra run triples that reviewers rejected as wrong. The draft
  shows the candidate with its evidence as input and ``{"keep_ids": []}`` as output.

Candidates are grouped by triple and ranked by how many runs made the error, so repeated errors
come first. The ``why`` of each draft is left for a person to write: an example teaches only
when it says what went wrong.

Leakage: examples must come from a development part of the gold set, never from the questions
results are reported on. Pass the development question ids as ``question_ids``; every
candidate carries its ``question_id`` so the split can be checked afterwards.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Collection, Iterable, Mapping
from typing import Any

from harness_schema.models import Run

from . import evaluate_gold_row
from .benchmark import PATHS, complexity_level, matching_runs
from .matching import Judge
from .relations import PredicateRelations

LOGGER = logging.getLogger(__name__)
EXTRACT_PROMPT = "extraction/edc_extract"
REVIEW_PROMPT = "extraction/evidence_review"
WHY_TODO = "TODO: say what the model got wrong and what to do instead"


def _words(predicate: str) -> str:
    """``hrn:evaluatesOn`` or a full IRI -> ``evaluates on`` (edc_extract uses plain relations)."""
    local = re.split(r"[#/:]", predicate)[-1] or predicate
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", local).replace("_", " ").casefold().strip()


def _extract_draft(item: dict[str, Any], question: str) -> dict[str, str]:
    source = item.get("source", "")
    evidence = item.get("evidence", "")
    proposition = {"evidence": evidence, "subject": item["s"], "predicate": _words(item["p"]),
                   "object": item["o"], "source_id": source}
    return {
        "input": f'question: "{question}"; passage {source}: "{evidence}"',
        "output": json.dumps({"triples": [proposition]}, ensure_ascii=False),
        "why": WHY_TODO,
    }


def _review_draft(item: dict[str, Any], question: str) -> dict[str, str]:
    candidate = f'0: ({item["s"]}, {item["p"]}, {item["o"]}) from {item.get("source", "?")}'
    if item.get("evidence"):
        candidate += f', evidence "{item["evidence"]}"'
    return {"input": f'question: "{question}"; candidate {candidate}',
            "output": json.dumps({"keep_ids": []}), "why": WHY_TODO}


def fewshot_candidates(
    candidates: Iterable[dict], runs: Iterable[Run], *, relations: PredicateRelations,
    judge: Judge | None = None, question_ids: Collection[str] | None = None,
    include_okay: bool = False, question_texts: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Missed gold triples and rejected extras over the chosen readings of eligible runs.

    A gold row's question id is its ``question_id`` (rows built by labelling), else the id the
    question library (``question_texts``: id -> text) gives its question text, else the row id.
    """
    runs = list(runs)
    by_text = {" ".join(text.split()).casefold(): qid for qid, text in (question_texts or {}).items()}
    grouped: dict[tuple, dict[str, Any]] = {}
    questions: set[str] = set()
    seen_ids: set[str] = set()
    scored_runs = 0
    for candidate in candidates:
        question = str(candidate.get("question", ""))
        question_id = str(candidate.get("question_id")
                          or by_text.get(" ".join(question.split()).casefold())
                          or candidate.get("id") or "")
        seen_ids.update({question_id, str(candidate.get("id") or "")})
        if question_ids is not None and question_id not in question_ids \
                and str(candidate.get("id") or "") not in question_ids:
            continue
        for path in PATHS:
            for run in matching_runs(candidate, runs, path):
                if run.status != "answered":
                    continue
                try:
                    result = evaluate_gold_row(run, candidate, relations=relations, judge=judge)
                except (ValueError, TypeError, KeyError) as exc:
                    LOGGER.warning("Cannot score run %s against %s: %s", run.id, candidate.get("id"), exc)
                    continue
                levels = result.get("chosen_levels")
                if not result["eligible_for_path_score"] or not levels:
                    continue
                scored_runs += 1
                questions.add(question_id)
                errors = [(EXTRACT_PROMPT, item) for item in levels["missed"]
                          if include_okay or item.get("weight") == "vital"]
                errors += [(REVIEW_PROMPT, item) for item in levels["extra"] if item.get("verdict") == "wrong"]
                for prompt, item in errors:
                    triple = {"s": item["s"], "p": item["p"], "o": item["o"]}
                    group = grouped.setdefault((prompt, question_id, triple["s"], triple["p"], triple["o"]), {
                        "prompt": prompt, "question_id": question_id, "question": question,
                        "complexity": complexity_level(candidate), "triple": triple,
                        "source": item.get("source"), "evidence": item.get("evidence"),
                        "weight": item.get("weight"), "errors": 0, "by_path": {}, "run_ids": [],
                        "draft": (_extract_draft if prompt == EXTRACT_PROMPT else _review_draft)(item, question),
                    })
                    group["errors"] += 1
                    shown = "adaptive-baseline" if run.path == "baseline-enhanced" else run.path
                    group["by_path"][shown] = group["by_path"].get(shown, 0) + 1
                    group["run_ids"].append(run.id)
    ranked = sorted(grouped.values(), key=lambda row: (row["prompt"], -row["errors"], row["question_id"]))
    return {
        "questions": sorted(questions),
        "scored_runs": scored_runs,
        "split": "development ids given" if question_ids is not None else "all approved questions",
        "unmatched_ids": sorted(set(question_ids or ()) - seen_ids),
        "candidates": {
            EXTRACT_PROMPT: [row for row in ranked if row["prompt"] == EXTRACT_PROMPT],
            REVIEW_PROMPT: [row for row in ranked if row["prompt"] == REVIEW_PROMPT],
        },
    }


__all__ = ["EXTRACT_PROMPT", "REVIEW_PROMPT", "fewshot_candidates"]
