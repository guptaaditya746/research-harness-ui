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
  says both state the same fact. Soft is symmetric. The rule edges are matched
  first; the judge is then asked about same-subject pairs that can still add
  credit (a leftover run triple against any gold triple, and a matched run
  triple against a leftover gold triple when its partner can move; see
  ``_ask_judge``).

What is counted is a fact, not a spelling: triples that entail each other under
the rules (an inverse restatement, a symmetric relation read backwards, a
same-as name) form one fact on each side, at every level. Euzenat's ratios are
taken over these facts, so restating a fact is neither an error nor a second
fact to recover, and stating everything twice cannot raise a score.

Credit is one-to-one: a maximum matching over the edges, so one gold fact is
never credited twice and one run fact never recovers two facts. This is
stricter than Euzenat's set-based Cn(R), where one run triple that implies two
gold triples (a relation and the type its domain gives) recovers both; here it
recovers one. Among equally large matchings the one covering more vital gold
triples, then the one with stricter pairs, is chosen (weighted assignment).

Other rules:

- **Locally closed DBLP scoring** (Text2KGBench): a run triple with a DBLP
  predicate the gold never uses for this question (nor a related one) is not
  scored. The gold lists only what the question needs, so an extra correct
  authorship triple is not an error. Text2KGBench closes every ground-truth
  relation this way; here only DBLP predicates are, because the invented
  relations are what the evaluation is about.
- **Vital recall**: recall over gold triples marked ``"weight": "vital"``
  (TREC 2024 RAG nugget evaluation). A fact is matched or not, so this is the
  track's strict variant (partial support earns nothing, not 0.5).
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


def _with_source(item: Mapping[str, Any]) -> dict[str, str]:
    """The triple plus its source id and a short evidence quote, when it has them (for error review)."""
    out = _plain(item)
    if item.get("source"):
        out["source"] = str(item["source"])
    evidence = str(item.get("evidence") or "").strip()
    if evidence and evidence != out.get("source"):
        out["evidence"] = evidence[:300]
    return out


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

    The objective is a sum over edges, so it is solved per connected component of the edge graph:
    the same optimum, without a dense |run| x |gold| matrix (D-sized inputs have thousands per side).
    """
    if not ranks or not rows or not cols:
        return {}
    parent: dict[tuple[str, int], tuple[str, int]] = {}

    def find(node: tuple[str, int]) -> tuple[str, int]:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for i, j in ranks:
        left, right = find(("r", i)), find(("g", j))
        if left != right:
            parent[right] = left
    components: dict[tuple[str, int], list[tuple[int, int]]] = {}
    for edge in ranks:
        components.setdefault(find(("r", edge[0])), []).append(edge)
    top = max(RANK.values())
    out: dict[int, int] = {}
    for edges in components.values():
        run_ids = sorted({i for i, _ in edges})
        gold_ids = sorted({j for _, j in edges})
        size = min(len(run_ids), len(gold_ids))
        vital_bonus = top * size + 1
        base = size * (vital_bonus + top) + 1
        row_of = {i: n for n, i in enumerate(run_ids)}
        col_of = {j: n for n, j in enumerate(gold_ids)}
        weights = np.zeros((len(run_ids), len(gold_ids)))
        for i, j in edges:
            weights[row_of[i], col_of[j]] = base + (vital_bonus if j in vital else 0) + ranks[(i, j)]
        row_ind, col_ind = linear_sum_assignment(weights, maximize=True)
        for r, c in zip(row_ind, col_ind, strict=True):
            i, j = run_ids[int(r)], gold_ids[int(c)]
            if (i, j) in ranks:
                out[i] = j
    return dict(sorted(out.items()))


def _ranked(*groups: tuple[set[tuple[int, int]], str]) -> dict[tuple[int, int], int]:
    ranks: dict[tuple[int, int], int] = {}
    for edges, kind in groups:
        for edge in edges:
            ranks[edge] = max(ranks.get(edge, -1), RANK[kind])
    return ranks


def _merge_gold(gold: Iterable[Triple | Mapping[str, Any]], key: Callable[[Mapping[str, Any]], Key]
                ) -> dict[Key, dict[str, Any]]:
    """Gold triples by key. Duplicates: vital if any copy is vital; their variants are pooled."""
    by_key: dict[Key, dict[str, Any]] = {}
    for item in map(_as_dict, gold):
        gold_key = key(item)
        previous = by_key.get(gold_key)
        if previous is None:
            by_key[gold_key] = item
            continue
        merged = dict(previous)
        if item.get("weight") == "vital":
            merged["weight"] = "vital"
        merged["variants"] = [*_variant_triples(previous),
                              *(v for v in _variant_triples(item) if v not in _variant_triples(previous))]
        by_key[gold_key] = merged
    return by_key


@dataclass
class _Pairs:
    """The run and gold facts of one scoring, indexed, with the rule edges between them.

    A fact is a class of triples that entail each other under the rules (an inverse restatement, a
    same-as name, a symmetric relation read backwards); ``run_keys``/``gold_keys`` hold one
    representative per class and ``run_members``/``gold_members`` every key in it.
    """

    run_by_key: dict[Key, dict[str, Any]]
    gold_by_key: dict[Key, dict[str, Any]]
    run_keys: list[Key]
    gold_keys: list[Key]
    excluded: list[Key]
    strict: set[tuple[int, int]]
    entailed_recall: set[tuple[int, int]]
    entailed_precision: set[tuple[int, int]]
    variants: set[tuple[int, int]]
    normal_run: list[Key]
    normal_gold: list[Key]
    run_members: list[list[Key]]
    gold_members: list[list[Key]]
    # Per fact: normalised subject -> the member stated from that subject (an inverse gives two).
    run_subjects: list[dict[str, Key]]
    gold_subjects: list[dict[str, Key]]

    @property
    def ruled(self) -> set[tuple[int, int]]:
        return self.strict | self.entailed_recall | self.entailed_precision | self.variants


def _equivalence_classes(keys: list[Key], closures: dict[Key, frozenset[Key]],
                         normal: dict[Key, Key]) -> list[list[Key]]:
    """Group keys whose triples entail each other (a ⊨ b and b ⊨ a), in first-seen order.

    With single-premise rules, mutual entailment is an equivalence and the members of a class have the
    same consequences, so one representative stands for the class at every level.
    """
    by_normal: dict[Key, list[Key]] = {}
    for item in keys:
        by_normal.setdefault(normal[item], []).append(item)
    parent = {item: item for item in keys}

    def find(item: Key) -> Key:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    order = {item: index for index, item in enumerate(keys)}
    for item in keys:
        for consequence in closures[item]:
            for other in by_normal.get(consequence, ()):
                if other != item and normal[item] in closures[other]:
                    left, right = find(item), find(other)
                    if left != right:
                        first, second = sorted((left, right), key=order.__getitem__)
                        parent[second] = first
    groups: dict[Key, list[Key]] = {}
    for item in keys:
        groups.setdefault(find(item), []).append(item)
    return list(groups.values())


def _merge_members(members: list[Key], items: Mapping[Key, Mapping[str, Any]]) -> dict[str, Any]:
    """One gold fact from its equivalent triples: vital if any is vital; every variant, as full triples."""
    merged = dict(items[members[0]])
    variants: list[dict[str, str]] = []
    for member in members:
        for variant in _variant_triples(items[member]):
            if variant not in variants:
                variants.append(variant)
    if variants:
        merged["variants"] = variants
    if any(items[member].get("weight") == "vital" for member in members):
        merged["weight"] = "vital"
    return merged


def _pairs(predicted: Iterable[Triple | Mapping[str, Any]], gold: Iterable[Triple | Mapping[str, Any]],
           key: Callable[[Mapping[str, Any]], Key], relations: PredicateRelations) -> _Pairs:
    merged_gold = _merge_gold(gold, key)
    run_items: dict[Key, dict[str, Any]] = {}
    for item in map(_as_dict, predicted):
        run_items.setdefault(key(item), item)

    gold_variants = {g: _variant_triples(item) for g, item in merged_gold.items()}
    # Locally closed: a DBLP-predicate run triple is scored only if the gold (or a variant) uses that
    # predicate or one related to it by a rule (parent, child, inverse).
    # A triple whose consequences reach such a predicate counts too (an inverse of a parent, say).
    gold_predicates = {g[1] for g in merged_gold} | {key(v)[1] for group in gold_variants.values() for v in group}
    allowed = relations.related_predicates(gold_predicates)
    excluded = [r for r, item in run_items.items() if r[1].startswith(DBLP_NS) and r[1] not in allowed
                and not {derived[1] for derived in relations.consequences(item, key, typing=False)} & allowed]
    excluded_set = set(excluded)
    kept = [r for r in run_items if r not in excluded_set]

    run_closure = {r: relations.consequences(run_items[r], key) for r in kept}
    gold_closure = {g: relations.consequences(item, key) for g, item in merged_gold.items()}
    run_normal = {r: relations.normalise(r) for r in kept}
    gold_normal = {g: relations.normalise(g) for g in merged_gold}
    run_members = _equivalence_classes(kept, run_closure, run_normal)
    gold_members = _equivalence_classes(list(merged_gold), gold_closure, gold_normal)
    run_keys = [members[0] for members in run_members]
    gold_keys = [members[0] for members in gold_members]
    # Every triple stays reachable by its key (members and excluded ones, for the report); the scored
    # facts are run_keys / gold_keys, and a gold representative carries its class's weight and variants.
    run_by_key = dict(run_items)
    gold_by_key = {**merged_gold, **{members[0]: _merge_members(members, merged_gold) for members in gold_members}}

    # Edges by index lookups (no |run| x |gold| scan): D can hold thousands of triples on each side.
    run_of = {member: i for i, members in enumerate(run_members) for member in members}
    run_of_normal: dict[Key, set[int]] = {}
    for i, r in enumerate(run_keys):
        run_of_normal.setdefault(run_normal[r], set()).add(i)
    gold_of_normal: dict[Key, set[int]] = {}
    for j, g in enumerate(gold_keys):
        gold_of_normal.setdefault(gold_normal[g], set()).add(j)
    strict = {(run_of[member], j) for j, members in enumerate(gold_members) for member in members
              if member in run_of}
    entailed_recall = {(i, j) for i, r in enumerate(run_keys) for c in run_closure[r]
                       for j in gold_of_normal.get(c, ())}
    entailed_precision = {(i, j) for j, g in enumerate(gold_keys) for c in gold_closure[g]
                          for i in run_of_normal.get(c, ())}
    variants = {(run_of[key(v)], j) for j, g in enumerate(gold_keys)
                for v in _variant_triples(gold_by_key[g]) if key(v) in run_of}
    return _Pairs(
        run_by_key, gold_by_key, run_keys, gold_keys, excluded,
        strict=strict, entailed_recall=entailed_recall, entailed_precision=entailed_precision, variants=variants,
        normal_run=[run_normal[r] for r in run_keys], normal_gold=[gold_normal[g] for g in gold_keys],
        run_members=run_members, gold_members=gold_members,
        run_subjects=[_by_subject(members, run_normal) for members in run_members],
        gold_subjects=[_by_subject(members, gold_normal) for members in gold_members],
    )


def _by_subject(members: list[Key], normal: Mapping[Key, Key]) -> dict[str, Key]:
    out: dict[str, Key] = {}
    for member in members:
        out.setdefault(normal[member][0], member)
    return out


def _rule_soft(pairs: _Pairs) -> dict[tuple[int, int], int]:
    return _ranked((pairs.variants, "variant"), (pairs.entailed_precision | pairs.entailed_recall, "entailed"),
                   (pairs.strict, "strict"))


def _vital(pairs: _Pairs) -> set[int]:
    return {j for j, g in enumerate(pairs.gold_keys) if pairs.gold_by_key[g].get("weight") == "vital"}


def _ask_judge(pairs: _Pairs, judge: Judge | None, vital: set[int] | None = None
               ) -> tuple[set[tuple[int, int]], int, int]:
    """Ask the judge about the pairs that can still add credit once the rules have matched what they can.

    The rule edges (strict, entailed, variant) are matched first. A judged edge can then add a pair in
    two ways: it joins a leftover run triple to a gold triple (a leftover, or one whose run partner can
    move), or it is the far end of an augmenting path of length three, ``r' - g0 = r0 - g1``: a leftover
    run triple ``r'`` takes ``g0`` and its partner ``r0`` takes the leftover ``g1``. So the judge is asked
    (1) every same-subject pair of a leftover run triple and any gold triple, then (2) the pairs of a
    matched run triple and a leftover gold triple, but only when the run triple's partner has a rule or
    judged edge to a leftover run triple. Longer augmenting paths (a judged edge between two matched
    triples) are not searched. Pairs with a rule edge are never asked.

    "Same subject" holds when any triple of the run fact and any triple of the gold fact share their
    normalised subject, so an inverse restatement is compared from the side the gold states it.
    Returns the judged matches and how many pairs were asked and are still pending.
    """
    judged: set[tuple[int, int]] = set()
    counts = {"asked": 0, "pending": 0}
    if judge is None:
        return judged, 0, 0
    ruled = pairs.ruled
    first = _matching(_rule_soft(pairs), len(pairs.run_keys), len(pairs.gold_keys),
                      _vital(pairs) if vital is None else vital)
    partner = {j: i for i, j in first.items()}
    leftover_runs = [i for i in range(len(pairs.run_keys)) if i not in first]
    leftover_gold = {j for j in range(len(pairs.gold_keys)) if j not in partner}
    gold_by_subject: dict[str, list[int]] = {}
    for j, subjects in enumerate(pairs.gold_subjects):
        for subject in subjects:
            gold_by_subject.setdefault(subject, []).append(j)

    def candidates(i: int, wanted: Callable[[int], bool]) -> list[tuple[int, Key, Key]]:
        out, seen = [], set()
        for subject, run_member in pairs.run_subjects[i].items():
            for j in gold_by_subject.get(subject, ()):
                if j not in seen and (i, j) not in ruled and wanted(j):
                    seen.add(j)
                    out.append((j, pairs.gold_subjects[j][subject], run_member))
        return out

    def ask(i: int, j: int, gold_member: Key, run_member: Key) -> None:
        counts["asked"] += 1
        # Both sides as stated from the shared subject (an inverse restatement is shown that way).
        verdict = judge(_plain(pairs.gold_by_key[gold_member]), _plain(pairs.run_by_key[run_member]))
        if verdict is None:
            counts["pending"] += 1
        elif verdict:
            judged.add((i, j))

    for i in leftover_runs:
        for j, gold_member, run_member in candidates(i, lambda _j: True):
            ask(i, j, gold_member, run_member)
    leftover_set = set(leftover_runs)
    movable = {j for i, j in ruled | judged if i in leftover_set and j in partner}
    for j0 in sorted(movable):
        i = partner[j0]
        for j, gold_member, run_member in candidates(i, leftover_gold.__contains__):
            ask(i, j, gold_member, run_member)
    return judged, counts["asked"], counts["pending"]


def _level_edges(pairs: _Pairs, judged: set[tuple[int, int]]
                 ) -> dict[str, tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]]]:
    """Per level, the (precision, recall) edges with their strength.

    Entailed is asymmetric (Euzenat): a run triple more specific than the gold recovers the gold
    fact, but the gold does not vouch for the extra detail, so it earns no precision credit.
    Soft is symmetric: it approximates a person asking "do these state the same fact?".
    """
    soft = _ranked((judged, "judge"), (pairs.variants, "variant"),
                   (pairs.entailed_precision | pairs.entailed_recall, "entailed"), (pairs.strict, "strict"))
    return {
        "strict": (_ranked((pairs.strict, "strict")),) * 2,
        "entailed": (_ranked((pairs.entailed_precision, "entailed"), (pairs.strict, "strict")),
                     _ranked((pairs.entailed_recall, "entailed"), (pairs.strict, "strict"))),
        "soft": (soft, soft),
    }


def _level_scores(pairs: _Pairs, edges: dict[str, tuple[dict, dict]], vital: set[int]
                  ) -> tuple[dict[str, Any], dict[str, tuple[dict[int, int], dict[int, int]]]]:
    rows, cols = len(pairs.run_keys), len(pairs.gold_keys)
    dblp_runs = {i for i, r in enumerate(pairs.run_keys) if r[1].startswith(DBLP_NS)}
    dblp_gold = {j for j, g in enumerate(pairs.gold_keys) if g[1].startswith(DBLP_NS)}
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
    return levels, matchings


def _judged_precision(pairs: _Pairs, judged: set[tuple[int, int]], rejected: Iterable[Mapping[str, Any]],
                      key: Callable[[Mapping[str, Any]], Key]) -> tuple[dict[str, Any], set[int]]:
    """The share of run triples known correct; returns it and the run indices known wrong."""
    # A rule link where only the run triple implies the gold (a more specific claim) is not verified
    # correctness: the gold does not vouch for the extra detail.
    correct = {i for i, _ in pairs.strict | pairs.entailed_precision | pairs.variants | judged}
    wrong_keys = {key(item) for item in map(_as_dict, rejected)
                  if str(item.get("verdict", "")).casefold() == "wrong"}
    wrong = {i for i, members in enumerate(pairs.run_members) if wrong_keys & set(members) and i not in correct}
    total = len(correct) + len(wrong)
    rows = len(pairs.run_keys)
    return {
        "value": len(correct) / total if total else None,
        "correct": len(correct), "wrong": len(wrong),
        "coverage": total / rows if rows else None,
    }, wrong


def _how(pairs: _Pairs, i: int, j: int) -> tuple[str, str]:
    if (i, j) in pairs.strict:
        return "strict", "canonical key"
    if (i, j) in pairs.entailed_recall or (i, j) in pairs.entailed_precision:
        return "entailed", "rule"
    if (i, j) in pairs.variants:
        return "soft", "variant"
    return "soft", "judge"


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
    pairs = _pairs(predicted, gold, key, relations or PredicateRelations())
    vital = _vital(pairs)
    judged, judge_asked, judge_pending = _ask_judge(pairs, judge, vital)
    levels, matchings = _level_scores(pairs, _level_edges(pairs, judged), vital)
    judged_precision, wrong_runs = _judged_precision(pairs, judged, rejected, key)

    credited_runs, recall_matching = matchings["soft"]
    run, gold_of = pairs.run_by_key, pairs.gold_by_key
    matched = []
    for i, j in sorted(recall_matching.items()):
        level, reason = _how(pairs, i, j)
        g = pairs.gold_keys[j]
        matched.append({"run": _plain(run[pairs.run_keys[i]]), "gold": _plain(gold_of[g]),
                        "level": level, "how": reason, "weight": gold_of[g].get("weight")})
    recalled_gold = set(recall_matching.values())
    missed = [{**_with_source(gold_of[g]), "weight": gold_of[g].get("weight")}
              for j, g in enumerate(pairs.gold_keys) if j not in recalled_gold]
    extra = [{**_with_source(run[r]), "verdict": "wrong" if i in wrong_runs else "unjudged"}
             for i, r in enumerate(pairs.run_keys) if i not in credited_runs]
    return {
        "levels": levels,
        "judged_precision": judged_precision,
        "judge": {"asked": judge_asked, "pending": judge_pending, "matched": len(judged)},
        "excluded_dblp": [_plain(run[r]) for r in pairs.excluded],
        # Triples that restate another one (inverse, same-as, symmetric): counted once, as one fact.
        "equivalent": {
            "run": [[_plain(run[member]) for member in members] for members in pairs.run_members
                    if len(members) > 1],
            "gold": [[_plain(gold_of[member]) for member in members] for members in pairs.gold_members
                     if len(members) > 1],
        },
        "matched": matched,
        "missed": missed,
        "extra": extra,
    }


__all__ = ["LEVELS", "Judge", "LevelScore", "score_levels"]
