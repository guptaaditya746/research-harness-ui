from harness_evaluation import evaluate_run, score_triples
from harness_schema.models import Metrics, Reading, Run, Triple


def triple(subject: str, predicate: str, object_: str) -> Triple:
    return Triple(s=subject, p=predicate, o=object_, support="direct", evidence="sentence", source="p1")


def test_score_triples_is_set_based() -> None:
    score = score_triples([triple("p1", "hrn:cites", "p2")], [triple("p1", "hrn:cites", "p2"), triple("p2", "hrn:cites", "p3")])
    assert score.as_dict() == {"predicted": 1, "gold": 2, "correct": 1, "precision": 1.0, "recall": 0.5, "f1": 2 / 3}


def test_evaluate_run_scores_selected_reading() -> None:
    reading = Reading(label="one", gloss="gloss", classes=[], axioms=[], sparql="SELECT * WHERE {}", answer="answer", triples=[triple("p1", "hrn:cites", "p2")], metrics=Metrics(), status="answered")
    run = Run(id="r-test", startedAt="2026-01-01T00:00:00Z", durationS=1, question="q", path="baseline", budget="fast", status="answered", readings=[reading], chosenReading=0)
    result = evaluate_run(run, [{"s": "p1", "p": "hrn:cites", "o": "p2"}])
    assert result["chosen"]["f1"] == 1.0
