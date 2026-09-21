import json
from pathlib import Path

from harness_baseline import Baseline20
from harness_corpus.store import CorpusStore
from harness_extraction import ExtractionEngine
from harness_schema.models import required_artifacts
from harness_semantic import hand_written_subschema


class RecordedModel:
    model = "recorded"

    def __init__(self) -> None:
        self.replies = iter(
            [
                {
                    "label": "Introduced method",
                    "gloss": "Paper proposing VPP",
                    "classes": ["Paper", "Claim"],
                    "properties": ["claims"],
                },
                {
                    "triples": [
                        {
                            "subject": "p4",
                            "predicate": "claims",
                            "object": "Video Prediction Policy",
                            "source_id": "p4-abstract",
                            "evidence": "we propose the Video Prediction Policy",
                        }
                    ]
                },
                {
                    "sparql": "PREFIX hrn: <https://harness.local/schema#>\nSELECT ?paper ?claim WHERE { ?paper hrn:claims ?claim . }",
                    "answer": "Paper p4 introduces Video Prediction Policy [p4-abstract].",
                    "cited_source_ids": ["p4-abstract"],
                },
            ]
        )

    def complete_json(self, system: str, user: str) -> dict:
        json.loads(user)
        return next(self.replies)


def corpus(tmp_path: Path) -> CorpusStore:
    rows = {
        "papers.jsonl": [{"paper_id": "p4", "title": "Video Prediction Policy", "year": 2025}],
        "passages.jsonl": [
            {
                "passage_id": "p4-abstract",
                "paper_id": "p4",
                "section_path": "Abstract",
                "para_idx": 0,
                "text": "We propose the Video Prediction Policy for robot learning.",
            }
        ],
        "facts.jsonl": [
            {
                "fact_id": "p4-metadata-0",
                "subject": "p4",
                "predicate": "title",
                "object": "Video Prediction Policy",
                "source_id": "p4-metadata",
            }
        ],
        "references.jsonl": [],
    }
    for name, values in rows.items():
        (tmp_path / name).write_text("".join(json.dumps(row) + "\n" for row in values))
    return CorpusStore(tmp_path)


def test_baseline_runs_four_nodes_and_writes_complete_artifact_set(tmp_path: Path) -> None:
    runner = Baseline20(
        corpus(tmp_path), ExtractionEngine(RecordedModel(), hand_written_subschema())
    )
    result = runner.run("Which paper introduced Video Prediction Policy?", run_id="r-live")

    assert result["run"].status == "answered"
    assert [row.action for row in result["run"].trace] == [
        "Retrieving top passages",
        "Inferring schema",
        "Extracting triples",
        "Querying graph and composing answer",
    ]
    assert set(result["artifacts"]) | {"run.json"} == set(required_artifacts("baseline"))
    assert "p4-abstract" in result["run"].readings[0].answer


def test_baseline_no_hits_is_a_terminal_failed_run(tmp_path: Path) -> None:
    runner = Baseline20(
        corpus(tmp_path), ExtractionEngine(RecordedModel(), hand_written_subschema())
    )
    result = runner.run("Unfindable zxqvterm", run_id="r-failed")

    assert result["run"].status == "incomplete"
    assert result["run"].reason == "no_results"
    assert set(result["artifacts"]) | {"run.json"} == set(required_artifacts("baseline"))


def test_enhanced_baseline_seeds_structural_facts_and_batches_extraction(tmp_path: Path) -> None:
    runner = Baseline20(
        corpus(tmp_path), ExtractionEngine(RecordedModel(), hand_written_subschema()), enhanced=True
    )
    result = runner.run("Which paper introduced Video Prediction Policy?", run_id="r-enhanced")

    assert result["run"].path == "baseline-enhanced"
    assert "retrieval.json" in result["artifacts"]
    assert "extraction_batches.json" in result["artifacts"]
    assert "corpus-seed" in result["artifacts"]["graph.trig"].decode()
    assert set(result["artifacts"]) | {"run.json"} == set(required_artifacts("baseline-enhanced"))
