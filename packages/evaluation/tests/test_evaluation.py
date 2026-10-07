import json

from harness_evaluation import evaluate_run, score_triples
from harness_evaluation.cli import evaluate_directory
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


def test_evaluate_directory_matches_dataset_question(tmp_path) -> None:
    run_dir = tmp_path / "artifacts" / "r-test"
    run_dir.mkdir(parents=True)
    reading = Reading(label="one", gloss="gloss", classes=[], axioms=[], sparql="SELECT * WHERE {}", answer="answer", triples=[triple("p1", "hrn:cites", "p2")], metrics=Metrics(), status="answered")
    run = Run(id="r-test", startedAt="2026-01-01T00:00:00Z", durationS=1, question="q", path="baseline", budget="fast", status="answered", readings=[reading], chosenReading=0)
    (run_dir / "run.json").write_text(run.model_dump_json(by_alias=True))
    dataset = tmp_path / "dev.jsonl"
    dataset.write_text(json.dumps({"question": "q", "gold_triples": [{"s": "p1", "p": "hrn:cites", "o": "p2"}]}) + "\n")
    result = evaluate_directory(tmp_path / "artifacts", dataset)
    assert result[0]["f1"] == 1.0
    # Reading-level fields come from the chosen reading (they used to read the triple score: always 0).
    assert result[0]["provenance_coverage"] == 1.0 and result[0]["not_in_gold_ratio"] == 0.0


def test_claims_are_aligned_optimally_not_greedily() -> None:
    from harness_evaluation import score_claims
    from harness_schema.models import ClaimSubgraph

    def claim(cid: str, **fields: str) -> ClaimSubgraph:
        return ClaimSubgraph.model_validate({
            "id": cid, "kind": "ResultClaim", "paperId": "p97", "sourceIds": ["p97-0"], "evidence": {"p97-0": "quote"},
            "fields": {key: {"value": value, "sourceIds": ["p97-0"]} for key, value in fields.items()}})

    # Greedy takes (A, X) first (2 equal fields) and leaves B with nothing; the optimum pairs A-Y, B-X.
    predicted = [claim("A", method="MethodX", dataset="BenchY", metric="acc"), claim("B", method="MethodX")]
    gold = [claim("X", method="MethodX", dataset="BenchY", metric="f1"), claim("Y", dataset="BenchY", metric="acc")]
    result = score_claims(predicted, gold)
    assert result["fields"]["correct"] == 3 and result["aligned"] == 2
