"""Triple precision and recall at three matching levels: strict, entailed, soft.

A run triple and a gold triple are joined by an edge when they state the same
fact at some level. The levels are cumulative (each keeps every edge of the
level before it):

- **strict**: equal canonical keys (``canonical_triple_key``: case, spacing,
  DBLP links, author names and predicate aliases are normalised).
- **entailed**: one triple logically implies the other under the RDFS/OWL rules
  in ``relations`` (sub-properties, inverses, class hierarchy, same-as).
  For recall the run triple must imply the gold triple; for precision the gold
  triple must imply the run triple (semantic precision/recall, Euzenat 2007).
  So a run triple more specific than the gold counts for recall only, and one
  more general than the gold counts for precision only.
- **soft**: a rule links the pair in either direction, the run triple equals
  one of the gold triple's accepted variants (typed by reviewers), or a judge
  says both state the same fact. Soft is symmetric. The judge is only asked
  about pairs with the same subject where neither triple has any rule or
  variant edge.

Credit is one-to-one: a maximum matching over the edges, so one gold triple is
never credited twice and one run triple never recovers two facts. Among equally
large matchings the one covering more vital gold triples, then the one with
stricter pairs, is chosen (weighted assignment).

Other rules:

- **Locally closed DBLP scoring** (Text2KGBench): a run triple with a DBLP
  predicate the gold never uses for this question (nor a related one) is not
  scored. The gold lists only what the question needs, so an extra correct
  authorship triple is not an error.
- **Vital recall**: recall over gold triples marked ``"weight": "vital"``
  (TREC 2024 RAG nugget evaluation).
- **Judged precision**: the share of run triples known to be correct, where
  correct means a strict, precision-side rule, variant or judge match to gold,
  and wrong means reviewers rejected it as wrong (``forbidden_alternatives``
  with ``verdict: wrong``). Reported with its coverage.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from harness_schema.models import Triple
from scipy.optimize import linear_sum_assignment

from .relations import DBLP_NS, PredicateRelations

LEVELS = ("strict", "entailed", "soft")
Key = tuple[str, str, str]
Judge = Callable[[Mapping[str, Any], Mapping[str, Any]], bool | None]


@dataclass(frozen=True)
class LevelScore:
    """Precision and recall when precision and recall can credit different pairs."""

    predicted: int
    gold: int
    matched_predicted: int
    matched_gold: int

    @property
    def precision(self) -> float:
        return self.matched_predicted / self.predicted if self.predicted else 0.0

    @property
    def recall(self) -> float:
        return self.matched_gold / self.gold if self.gold else 0.0

    @property
    def f1(self) -> float:
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {"predicted": self.predicted, "gold": self.gold,
                "matched_predicted": self.matched_predicted, "matched_gold": self.matched_gold,
                "precision": self.precision, "recall": self.recall, "f1": self.f1}


def _as_dict(value: Triple | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, Triple):
        return value.model_dump(by_alias=True)
    return dict(value)


def _plain(item: Mapping[str, Any]) -> dict[str, str]:
    return {"s": str(item.get("s", "")), "p": str(item.get("p", "")), "o": str(item.get("o", ""))}


def _variant_triples(gold: Mapping[str, Any]) -> list[dict[str, str]]:
    """Variants are full triples, or just another wording of the object."""
    out = []
    raw = gold.get("variants") or []
    for variant in [raw] if isinstance(raw, str | Mapping) else raw:
        if isinstance(variant, str):
            if variant.strip():
                out.append({"s": str(gold.get("s", "")), "p": str(gold.get("p", "")), "o": variant.strip()})
        elif isinstance(variant, Mapping):
            out.append({"s": str(variant.get("s") or gold.get("s", "")),
                        "p": str(variant.get("p") or gold.get("p", "")),
                        "o": str(variant.get("o", ""))})
    return [item for item in out if item["o"]]


# Edge strength, used to prefer stricter matches among equally large matchings.
RANK = {"judge": 0, "variant": 1, "entailed": 2, "strict": 3}


def _matching(ranks: dict[tuple[int, int], int], rows: int, cols: int, vital: set[int]) -> dict[int, int]:
    """One-to-one credit: run index -> gold index.

    A weighted assignment (SciPy's ``linear_sum_assignment``) whose weights make, in this order,
    (1) more pairs, (2) more vital gold triples and (3) stricter edges always win. So the number of
    pairs equals a maximum bipartite matching, vital recall never drops at a looser level, and each
    pair is reported at the strictest level available.
    """
    if not ranks or not rows or not cols:
        return {}
    size = min(rows, cols)
    vital_bonus = max(RANK.values()) * size + 1
    base = size * (vital_bonus + max(RANK.values())) + 1
    weights = np.zeros((rows, cols))
    for (i, j), rank in ranks.items():
        weights[i, j] = base + (vital_bonus if j in vital else 0) + rank
    row_ind, col_ind = linear_sum_assignment(weights, maximize=True)
    return {int(i): int(j) for i, j in zip(row_ind, col_ind, strict=True) if (int(i), int(j)) in ranks}


def _ranked(*groups: tuple[set[tuple[int, int]], str]) -> dict[tuple[int, int], int]:
    ranks: dict[tuple[int, int], int] = {}
    for edges, kind in groups:
        for edge in edges:
            ranks[edge] = max(ranks.get(edge, -1), RANK[kind])
    return ranks


def score_levels(
    predicted: Iterable[Triple | Mapping[str, Any]],
    gold: Iterable[Triple | Mapping[str, Any]],
    *,
    key: Callable[[Mapping[str, Any]], Key],
    relations: PredicateRelations | None = None,
    judge: Judge | None = None,
    rejected: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Score one reading's triples against one gold set at every level."""
    relations = relations or PredicateRelations()
    gold_by_key: dict[Key, dict[str, Any]] = {}
    for item in map(_as_dict, gold):
        gold_key = key(item)
        previous = gold_by_key.get(gold_key)
        if previous is None:
            gold_by_key[gold_key] = item
            continue
        # Duplicates: vital if any copy is vital; their variants are pooled.
        merged = dict(previous)
        if item.get("weight") == "vital":
            merged["weight"] = "vital"
        merged["variants"] = [*_variant_triples(previous),
                              *(v for v in _variant_triples(item) if v not in _variant_triples(previous))]
        gold_by_key[gold_key] = merged
    run_by_key: dict[Key, dict[str, Any]] = {}
    for item in map(_as_dict, predicted):
        run_by_key.setdefault(key(item), item)

    gold_keys = list(gold_by_key)
    variant_triples = [_variant_triples(gold_by_key[g]) for g in gold_keys]
    # Locally closed: a DBLP-predicate run triple is scored only if the gold (or a variant) uses that
    # predicate or one related to it by a rule (parent, child, inverse).
    # A triple whose consequences reach such a predicate counts too (an inverse of a parent, say).
    gold_predicates = {g[1] for g in gold_keys} | {key(v)[1] for group in variant_triples for v in group}
    allowed = relations.related_predicates(gold_predicates)
    closure_by_key = {r: relations.consequences(item, key) for r, item in run_by_key.items()}
    excluded = [r for r in run_by_key if r[1].startswith(DBLP_NS) and r[1] not in allowed
                and not {derived[1] for derived in closure_by_key[r]} & allowed]
    excluded_set = set(excluded)
    run_keys = [r for r in run_by_key if r not in excluded_set]

    run_closure = [closure_by_key[r] for r in run_keys]
    gold_closure = [relations.consequences(gold_by_key[g], key) for g in gold_keys]
    normal_run = [relations.normalise(r) for r in run_keys]
    normal_gold = [relations.normalise(g) for g in gold_keys]

    strict = {(i, j) for i, r in enumerate(run_keys) for j, g in enumerate(gold_keys) if r == g}
    entailed_recall = {(i, j) for i in range(len(run_keys)) for j in range(len(gold_keys))
                       if normal_gold[j] in run_closure[i]}
    entailed_precision = {(i, j) for i in range(len(run_keys)) for j in range(len(gold_keys))
                          if normal_run[i] in gold_closure[j]}
    variant_keys = [{key(item) for item in group} for group in variant_triples]
    variants = {(i, j) for i, r in enumerate(run_keys) for j in range(len(gold_keys)) if r in variant_keys[j]}

    ruled = strict | entailed_recall | entailed_precision | variants
    linked_runs = {i for i, _ in ruled}
    linked_gold = {j for _, j in ruled}
    judged: set[tuple[int, int]] = set()
    judge_pending = 0
    judge_asked = 0
    if judge is not None:
        for i, r in enumerate(run_keys):
            if i in linked_runs:
                continue
            for j, g in enumerate(gold_keys):
                if j in linked_gold or normal_run[i][0] != normal_gold[j][0]:
                    continue
                judge_asked += 1
                verdict = judge(_plain(gold_by_key[g]), _plain(run_by_key[r]))
                if verdict is None:
                    judge_pending += 1
                elif verdict:
                    judged.add((i, j))

    # Entailed is asymmetric (Euzenat): a run triple more specific than the gold recovers the gold
    # fact, but the gold does not vouch for the extra detail, so it earns no precision credit.
    # Soft is symmetric: it approximates a person asking "do these state the same fact?".
    soft = _ranked((judged, "judge"), (variants, "variant"), (entailed_precision | entailed_recall, "entailed"),
                   (strict, "strict"))
    edges = {
        "strict": (_ranked((strict, "strict")),) * 2,
        "entailed": (_ranked((entailed_precision, "entailed"), (strict, "strict")),
                     _ranked((entailed_recall, "entailed"), (strict, "strict"))),
        "soft": (soft, soft),
    }
    rows, cols = len(run_keys), len(gold_keys)
    vital = {j for j, g in enumerate(gold_keys) if gold_by_key[g].get("weight") == "vital"}
    dblp_runs = {i for i, r in enumerate(run_keys) if r[1].startswith(DBLP_NS)}
    dblp_gold = {j for j, g in enumerate(gold_keys) if g[1].startswith(DBLP_NS)}
    levels: dict[str, Any] = {}
    matchings: dict[str, tuple[dict[int, int], dict[int, int]]] = {}
    for level, (precision_edges, recall_edges) in edges.items():
        by_precision = _matching(precision_edges, rows, cols, vital)
        by_recall = _matching(recall_edges, rows, cols, vital)
        matchings[level] = (by_precision, by_recall)
        recalled = set(by_recall.values())
        credited = set(by_precision)
        levels[level] = LevelScore(rows, cols, len(credited), len(recalled)).as_dict()
        levels[level]["vital_recall"] = len(recalled & vital) / len(vital) if vital else None
        levels[level]["parts"] = {
            "dblp": LevelScore(len(dblp_runs), len(dblp_gold), len(credited & dblp_runs),
                               len(recalled & dblp_gold)).as_dict(),
            "invented": LevelScore(rows - len(dblp_runs), cols - len(dblp_gold),
                                   len(credited - dblp_runs), len(recalled - dblp_gold)).as_dict(),
        }

    # A rule link where only the run triple implies the gold (a more specific claim) is not verified
    # correctness: the gold does not vouch for the extra detail.
    correct_runs = {i for i, _ in strict | entailed_precision | variants | judged}
    wrong_keys = {key(item) for item in map(_as_dict, rejected)
                  if str(item.get("verdict", "")).casefold() == "wrong"}
    wrong_runs = {i for i, r in enumerate(run_keys) if r in wrong_keys and i not in correct_runs}
    judged_total = len(correct_runs) + len(wrong_runs)
    judged_precision = {
        "value": len(correct_runs) / judged_total if judged_total else None,
        "correct": len(correct_runs), "wrong": len(wrong_runs),
        "coverage": judged_total / rows if rows else None,
    }

    def how(i: int, j: int) -> tuple[str, str]:
        if (i, j) in strict:
            return "strict", "canonical key"
        if (i, j) in entailed_recall or (i, j) in entailed_precision:
            return "entailed", "rule"
        if (i, j) in variants:
            return "soft", "variant"
        return "soft", "judge"

    _, recall_matching = matchings["soft"]
    matched = []
    for i, j in sorted(recall_matching.items()):
        level, reason = how(i, j)
        matched.append({"run": _plain(run_by_key[run_keys[i]]), "gold": _plain(gold_by_key[gold_keys[j]]),
                        "level": level, "how": reason, "weight": gold_by_key[gold_keys[j]].get("weight")})
    recalled_gold = set(recall_matching.values())
    missed = [{**_plain(gold_by_key[g]), "weight": gold_by_key[g].get("weight")}
              for j, g in enumerate(gold_keys) if j not in recalled_gold]
    credited_runs = set(matchings["soft"][0])
    extra = [{**_plain(run_by_key[r]), "verdict": "wrong" if i in wrong_runs else "unjudged"}
             for i, r in enumerate(run_keys) if i not in credited_runs]
    return {
        "levels": levels,
        "judged_precision": judged_precision,
        "judge": {"asked": judge_asked, "pending": judge_pending, "matched": len(judged)},
        "excluded_dblp": [_plain(run_by_key[r]) for r in excluded],
        "matched": matched,
        "missed": missed,
        "extra": extra,
    }


__all__ = ["LEVELS", "Judge", "LevelScore", "score_levels"]
