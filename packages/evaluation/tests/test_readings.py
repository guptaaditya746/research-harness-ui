"""Ambiguous questions: AmbigQA set F1 over the union of the accepted readings, and attempts.

Made-up entities (AuthorA, AuthorB, p98, p99); no evaluation question is used.
"""

from __future__ import annotations

import pytest
from harness_evaluation import evaluate_attempt, evaluate_gold_row, row_gold
from harness_evaluation.benchmark import build_benchmark, group_attempts
from harness_evaluation.relations import PredicateRelations
from harness_schema.models import ConceptChoice, Metrics, Reading, Run

RELATIONS = PredicateRelations()


def t(s: str, p: str, o: str, **extra) -> dict:
    return {"s": s, "p": p, "o": o, "support": "test", "evidence": "e", "source": "x", **extra}


MOST_PAPERS = [t("AuthorA", "hasPaperCount", "12", weight="vital")]
MOST_CITED = [t("AuthorB", "hasCitationCount", "900", weight="vital")]
ROW = {"id": "g-amb", "question": "Who is the strongest author here?", "gold_triples": [],
       "complexity": {"level": "complex"},
       "accepted_readings": [{"label": "most papers", "measure": "papers", "gold_triples": MOST_PAPERS},
                             {"label": "most cited", "measure": "citations", "gold_triples": MOST_CITED}]}


def run(run_id: str, triples: list[dict], *, label: str = "reading", attempt: str | None = None,
        status: str = "answered", started: str = "2026-01-01T00:00:00Z", question: str = ROW["question"]) -> Run:
    reading = Reading(label=label, gloss="g", classes=[], axioms=[], sparql="", answer="a",
                      triples=[{k: v for k, v in item.items() if k not in {"weight", "variants"}} for item in triples],
                      metrics=Metrics(), status="answered")
    extra = {} if status == "answered" else {"reason": "no_results"}
    return Run(id=run_id, startedAt=started, durationS=1, question=question, path="agentic",
               budget="fast", status=status, readings=[reading], chosenReading=0, attemptId=attempt,
               concept=ConceptChoice(label=label, gloss="g"), **extra)


def soft(result: dict) -> dict:
    return result["chosen_levels"]["levels"]["soft"]


def test_one_of_two_readings_gets_half_recall_and_both_get_full_recall() -> None:
    one = evaluate_gold_row(run("r1", MOST_PAPERS), ROW, relations=RELATIONS)
    both = evaluate_gold_row(run("r2", MOST_PAPERS + MOST_CITED), ROW, relations=RELATIONS)
    assert (soft(one)["precision"], soft(one)["recall"]) == (1.0, 0.5)
    assert soft(one)["f1"] == pytest.approx(2 / 3) and soft(one)["vital_recall"] == 0.5
    assert (soft(both)["precision"], soft(both)["recall"], soft(both)["f1"]) == (1.0, 1.0, 1.0)
    # The exact metric uses the same union.
    assert (one["chosen"]["recall"], both["chosen"]["recall"]) == (0.5, 1.0)
    assert one["levels_gold"] == "readings_union" and one["gold"] == 2
    assert "accepted_reading" not in one                     # no best-of-readings selection any more


def test_per_reading_recall_and_readings_covered() -> None:
    one = evaluate_gold_row(run("r1", MOST_PAPERS), ROW, relations=RELATIONS)
    assert one["readings_recall"] == {"most papers": 1.0, "most cited": 0.0}
    assert (one["readings_covered"], one["readings_accepted"]) == (1, 2)
    plain = evaluate_gold_row(run("r2", MOST_PAPERS), {"gold_triples": MOST_PAPERS}, relations=RELATIONS)
    assert (plain["levels_gold"], plain["readings_recall"], plain["readings_covered"]) == ("row", {}, None)


def test_the_union_counts_a_shared_triple_once_and_includes_the_row_s_own_gold() -> None:
    shared = t("p99", "authoredBy", "AuthorA")
    row = {"gold_triples": [t("p98", "title", "Paper Z")],
           "accepted_readings": [{"label": "x", "gold_triples": [shared, *MOST_PAPERS]},
                                 {"label": "y", "gold_triples": [{**shared, "weight": "vital"}]},
                                 {"label": "no gold", "gold_triples": []}]}
    assert len(row_gold(row)) == 4                           # the reading without gold adds nothing
    result = evaluate_gold_row(run("r", [shared]), row, relations=RELATIONS)
    assert soft(result)["gold"] == 3                         # title, shared (once), paper count
    assert soft(result)["recall"] == pytest.approx(1 / 3)
    assert soft(result)["vital_recall"] == 0.5               # shared is vital (from y), paper count is vital
    assert result["readings_recall"] == {"x": 0.5, "y": 1.0} and result["readings_covered"] == 2


def test_an_attempt_unions_its_reading_runs_into_one_prediction() -> None:
    runs = [run("a1", MOST_PAPERS, label="most papers", attempt="a1"),
            run("a2", MOST_CITED, label="most cited", attempt="a1")]
    result = evaluate_attempt(runs, ROW, relations=RELATIONS)
    assert (soft(result)["recall"], soft(result)["f1"]) == (1.0, 1.0)
    assert result["readings_covered"] == 2 and result["scored_runs"] == ["a1", "a2"]
    # A failed reading run adds nothing: recall falls to the readings that were answered.
    failed = [runs[0], run("a3", [], label="most cited", attempt="a1", status="incomplete")]
    partial = evaluate_attempt(failed, ROW, relations=RELATIONS)
    assert soft(partial)["recall"] == 0.5 and partial["eligible_for_path_score"]
    # A reading researched twice in one attempt counts once (its latest eligible run).
    again = [*runs, run("a4", [t("AuthorC", "hasCitationCount", "5")], label="most cited", attempt="a1",
                        started="2026-01-02T00:00:00Z")]
    redone = evaluate_attempt(again, ROW, relations=RELATIONS)
    assert redone["scored_runs"] == ["a1", "a4"] and soft(redone)["recall"] == 0.5
    assert not evaluate_attempt([failed[1]], ROW, relations=RELATIONS)["eligible_for_path_score"]


def test_runs_group_by_attempt_id_and_the_run_whose_id_they_share_joins_them() -> None:
    runs = [run("p1", [], attempt=None), run("c1", [], attempt="p1"), run("x", []), run("c2", [], attempt="q")]
    # Reading runs recorded before attempt ids existed share their proposal as parent.
    legacy = [run("l1", []).model_copy(update={"parent_run_id": "prop"}),
              run("l2", []).model_copy(update={"parent_run_id": "prop"})]
    assert [[item.id for item in group] for group in group_attempts([*runs, *legacy])] == [
        ["p1", "c1"], ["x"], ["c2"], ["l1", "l2"]]


def test_benchmark_scores_an_attempt_once_and_never_averages_reading_runs_as_repetitions() -> None:
    attempt = [run("a1", MOST_PAPERS, label="most papers", attempt="att-1"),
               run("a2", MOST_CITED, label="most cited", attempt="att-1")]
    loose = [run("b1", MOST_PAPERS, label="most papers"), run("b2", MOST_CITED, label="most cited")]
    for runs, attempts, f1, covered in [(attempt, 1, 1.0, 2), (loose, 2, 2 / 3, 1)]:
        result = build_benchmark([ROW], runs, relations=RELATIONS)
        entry = result["questions"][0]["paths"][0]
        assert entry["attempted"] == entry["completed"] == attempts
        assert entry["attempts"] == ([["a1", "a2"]] if attempts == 1 else [["b1"], ["b2"]])
        assert entry["levels"]["soft"]["f1"]["mean"] == pytest.approx(f1)
        assert entry["readings_covered"]["mean"] == covered
    loose_entry = build_benchmark([ROW], loose, relations=RELATIONS)["questions"][0]["paths"][0]
    assert loose_entry["readings_recall"] == {"most papers": 0.5, "most cited": 0.5}


def test_a_failed_attempt_scores_zero_recall_on_every_reading() -> None:
    runs = [run("f1", [], label="most papers", attempt="att", status="incomplete"),
            run("f2", [], label="most cited", attempt="att", status="incomplete")]
    entry = build_benchmark([ROW], runs, relations=RELATIONS)["questions"][0]["paths"][0]
    assert (entry["attempted"], entry["failed"]) == (1, 1)
    assert entry["readings_recall"] == {"most papers": 0.0, "most cited": 0.0}
    assert entry["levels"]["soft"]["f1"]["mean"] == 0.0
