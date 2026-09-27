"""Few-shot candidates: missed vital gold triples and reviewer-rejected extras, repeated errors first."""

from __future__ import annotations

import json

from harness_evaluation.fewshot import EXTRACT_PROMPT, REVIEW_PROMPT, fewshot_candidates
from harness_evaluation.relations import PredicateRelations
from harness_schema.models import Metrics, Reading, Run

PAPER = "https://dblp.org/rec/conf/acl/SafeRAG25"


def t(s: str, p: str, o: str, **extra) -> dict:
    return {"s": s, "p": p, "o": o, "support": "test", "evidence": "e", "source": "x", **extra}


def run(run_id: str, path: str, question: str, triples: list[dict]) -> Run:
    plain = [{k: v for k, v in item.items() if k != "weight"} for item in triples]
    reading = Reading(label="one", gloss="g", classes=[], axioms=[], sparql="", answer="a",
                      triples=plain, metrics=Metrics(), status="answered")
    return Run(id=run_id, startedAt="2026-01-01T00:00:00Z", durationS=1, question=question, path=path,
               budget="fast", status="answered", readings=[reading], chosenReading=0)


BENCHMARK = t(PAPER, "hrn:contributionType", "Benchmark", weight="vital", source="saferag-abs-1",
              evidence="we propose SafeRAG, a benchmark designed to evaluate RAG security")
ATTACK = t(PAPER, "evaluatesAttack", "soft ad", weight="okay", source="saferag-3-2", evidence="soft ad attack")
WRONG = t(PAPER, "addressesProblem", "prompt injection", source="saferag-1-1", evidence="injection")
GOLD_ROW = {"id": "gold-v1-nlp27", "question_id": "nlp27", "question": "Q27", "status": "approved",
            "gold_triples": [BENCHMARK, ATTACK], "complexity": {"level": "compound"},
            "forbidden_alternatives": [{**WRONG, "verdict": "wrong"}]}
OTHER_ROW = {"id": "gold-v1-nlp05", "question_id": "nlp05", "question": "Q5", "status": "approved",
             "gold_triples": [t(PAPER, "title", "SafeRAG", weight="vital")]}


def test_candidates_rank_repeated_errors_and_keep_the_evidence() -> None:
    runs = [
        run("r1", "baseline", "Q27", [ATTACK, WRONG]),
        run("r2", "agentic", "Q27", [ATTACK]),
        run("r3", "baseline-enhanced", "Q27", [BENCHMARK]),
        run("r4", "baseline", "Q5", []),
    ]
    result = fewshot_candidates([GOLD_ROW, OTHER_ROW], runs, relations=PredicateRelations())
    assert result["questions"] == ["nlp05", "nlp27"] and result["scored_runs"] == 4
    extract = result["candidates"][EXTRACT_PROMPT]
    # Benchmark was missed twice (r1, r2), the nlp05 title once; the okay triple is not listed.
    assert [(row["question_id"], row["triple"]["o"], row["errors"]) for row in extract] == [
        ("nlp27", "Benchmark", 2), ("nlp05", "SafeRAG", 1)]
    first = extract[0]
    assert first["by_path"] == {"baseline": 1, "agentic": 1} and first["run_ids"] == ["r1", "r2"]
    assert first["source"] == "saferag-abs-1" and first["evidence"].startswith("we propose SafeRAG")
    proposition = json.loads(first["draft"]["output"])["triples"][0]
    assert proposition["predicate"] == "contribution type" and proposition["source_id"] == "saferag-abs-1"
    assert "saferag-abs-1" in first["draft"]["input"] and first["draft"]["why"].startswith("TODO")
    review = result["candidates"][REVIEW_PROMPT]
    assert [(row["triple"]["o"], row["errors"], row["run_ids"]) for row in review] == [
        ("prompt injection", 1, ["r1"])]
    assert json.loads(review[0]["draft"]["output"])["keep_ids"] == []


def test_development_ids_restrict_the_questions_and_okay_triples_are_optional() -> None:
    runs = [run("r1", "baseline", "Q27", []), run("r4", "baseline", "Q5", [])]
    result = fewshot_candidates([GOLD_ROW, OTHER_ROW], runs, relations=PredicateRelations(),
                                question_ids={"nlp27"}, include_okay=True)
    assert result["questions"] == ["nlp27"] and result["split"] == "development ids given"
    assert sorted(row["triple"]["o"] for row in result["candidates"][EXTRACT_PROMPT]) == ["Benchmark", "soft ad"]


def test_older_gold_rows_are_named_by_the_question_library() -> None:
    row = {**OTHER_ROW, "id": "gold-746fc49e48"}
    del row["question_id"]
    runs = [run("r4", "baseline", "Q5", [])]
    result = fewshot_candidates([row], runs, relations=PredicateRelations(), question_ids={"nlp05", "nlp99"},
                                question_texts={"nlp05": " q5 "})
    assert result["questions"] == ["nlp05"] and result["unmatched_ids"] == ["nlp99"]
