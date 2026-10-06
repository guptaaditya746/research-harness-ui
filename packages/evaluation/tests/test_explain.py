"""``make evaluate --explain``: the reasons a gold row scores zero or nothing, in plain language."""

from harness_evaluation.explain import explain
from harness_evaluation.relations import PredicateRelations
from harness_schema.models import EvaluationSnapshot, Metrics, Reading, Run, Triple

SNAPSHOT = {"corpusSnapshotId": "c1", "ontologyVersion": "o1", "claimSchemaVersion": "v3", "goldVersion": None,
            "retrievalIndexVersion": "i1"}
DBLP = "https://dblp.org/rec/conf/x/MethodX25"


def run(triples: list[tuple[str, str, str]], *, question: str = "Which benchmarks does MethodX use?",
        snapshot: dict = SNAPSHOT, status: str = "answered", run_id: str = "r-1") -> Run:
    reading = Reading(label="one", gloss="", classes=[], axioms=[], sparql="", answer="a", metrics=Metrics(),
                      status="answered", triples=[Triple(s=s, p=p, o=o, support="1 passage", evidence="e",
                                                         source="p99-1") for s, p, o in triples])
    return Run(id=run_id, startedAt="2026-01-01T00:00:00Z", durationS=1, question=question, path="baseline",
               budget="fast", status=status, readings=[reading], chosenReading=0,
               snapshot=EvaluationSnapshot.model_validate(snapshot),
               reason=None if status == "answered" else "deadline")


def row(subject: str = "p99") -> dict:
    return {"id": "gold-v1-q1", "question": "Which benchmarks does MethodX use?", "gold_version": "v1-draft",
            "snapshot": SNAPSHOT, "gold_triples": [{"s": subject, "p": "evaluatesOn", "o": "BenchY"}]}


def hints(gold: dict, runs: list[Run], judge=None) -> str:
    return " | ".join(explain(gold, runs, relations=PredicateRelations(), judge=judge)["hints"])


def test_pairing_problems_are_named() -> None:
    assert "no saved run has this exact question text" in hints(row(), [run([], question="Which benchmark?")])
    text = hints(row(), [run([(DBLP, "evaluatesOn", "BenchY")], snapshot={**SNAPSHOT, "ontologyVersion": "o2"})])
    assert "another evaluation snapshot" in text and "'ontologyVersion': ['o1', 'o2']" in text
    assert "only answered runs are scored" in hints(row(), [run([], status="incomplete")])


def test_local_paper_keys_against_dblp_iris_share_no_subject() -> None:
    text = hints(row("p99"), [run([(DBLP, "evaluatesOn", "BenchY")])])
    assert "share no subject" in text and "local key" in text
    report = explain(row(DBLP), [run([(DBLP, "evaluatedOn", "BenchY")])], relations=PredicateRelations(),
                     judge=lambda gold, candidate: None)
    comparison = report["paths"]["baseline"]["first_answered"]
    assert comparison["shared_subjects"] == 1 and comparison["strict_pairs"] == 0
    assert comparison["judge_pairs"] == 1 and comparison["judge_pending"] == 1
    assert any("make evaluate ARGS=--judge" in hint for hint in report["hints"])
    assert any("strict is 0 by construction" in hint for hint in report["hints"])
