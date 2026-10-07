"""Fast/Adaptive retrieval: one code path; HARNESS_RETRIEVAL_MODE only adds the dense list to the fusion."""

from __future__ import annotations

import json
import time
from pathlib import Path

import bm25s
import numpy as np
import pytest
from harness_baseline import Baseline20
from harness_baseline.planner import QueryPlan
from harness_corpus import store as store_module
from harness_corpus.dense import passages_digest
from harness_corpus.store import CorpusStore
from harness_extraction import ExtractionEngine
from harness_semantic import hand_written_subschema

TITLES = {"p1": "MethodX Prime Engine", "p2": "BenchY Suite", "p3": "Pruning Study", "p4": "Bread Notes"}
PASSAGES = {
    "p1-a": "MethodX compresses sparse attention for long documents and reduces memory.",
    "p1-b": "Sparse attention in MethodX keeps accuracy on long documents.",
    "p1-c": "Limitations: MethodX needs a tuned block size.",
    "p1-d": "We thank the reviewers for their comments.",
    "p2-a": "BenchY measures sparse attention accuracy on long documents across tasks.",
    "p2-b": "BenchY reports memory use of attention.",
    "p3-a": "Attention heads can be pruned after training.",
    "p3-b": "Documents are split into chunks before encoding.",
    "p4-a": "A recipe for bread with long fermentation.",
}
QUESTION = "How does sparse attention handle long documents?"


class NoModel:
    model = "none"

    def complete_json(self, system: str, user: str) -> dict:
        raise AssertionError("retrieval calls no model")


def _corpus(root: Path, *, dense: bool = False) -> CorpusStore:
    root.mkdir(parents=True, exist_ok=True)
    papers = [{"paper_id": pid, "title": title, "year": 2025, "abstract": ""} for pid, title in TITLES.items()]
    passages = [{"passage_id": pid, "paper_id": pid.split("-")[0], "section_path": "Body", "para_idx": i,
                 "text": text} for i, (pid, text) in enumerate(PASSAGES.items())]
    for name, rows in (("papers.jsonl", papers), ("passages.jsonl", passages), ("facts.jsonl", []),
                       ("references.jsonl", [])):
        (root / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    folder = root / "index" / "bm25"
    folder.mkdir(parents=True, exist_ok=True)
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize([p["text"] for p in passages], stopwords="en", show_progress=False),
                    show_progress=False)
    retriever.save(str(folder), corpus=[{"passage_id": p["passage_id"]} for p in passages], show_progress=False)
    if dense:
        vectors = root / "index" / "dense"
        vectors.mkdir(parents=True, exist_ok=True)
        # The dense ranking prefers the pruning and bread passages: it disagrees with BM25.
        matrix = np.asarray([[1.0, 0.1] if pid in {"p3-a", "p4-a"} else [0.1, 1.0] for pid in PASSAGES],
                            dtype=np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        np.save(vectors / "embeddings.npy", matrix.astype(np.float16))
        (vectors / "passage_ids.json").write_text(json.dumps(list(PASSAGES)))
        (vectors / "meta.json").write_text(json.dumps({"model": "test", "passages_sha256": passages_digest(root)}))
    store_module._embed_query_cached.cache_clear()
    store_module._embedding_down_until.clear()
    return CorpusStore(root)


def _retrieve(store: CorpusStore, *, question: str = QUESTION, enhanced: bool = False,
              max_passages: int | None = None, retrieve_k: int = 3, plan: QueryPlan | None = None,
              paper_ids: list[str] | None = None) -> dict:
    runner = Baseline20(store, ExtractionEngine(NoModel(), hand_written_subschema()), retrieve_k=retrieve_k,
                        enhanced=enhanced, max_passages=max_passages)
    runner.started = time.monotonic()
    return runner._retrieve({"question": question, "plan": plan or QueryPlan(operation="research"),
                             "paper_ids": paper_ids or [], "concept": None})


def _ids(result: dict) -> list[str]:
    return [str(row["passage_id"]) for row in result["passages"]]


@pytest.mark.parametrize(("enhanced", "max_passages"), [(False, None), (True, 2), (True, 5)])
def test_lexical_and_hybrid_select_the_same_passages_without_a_dense_index(
        tmp_path: Path, monkeypatch, enhanced: bool, max_passages: int | None) -> None:
    store = _corpus(tmp_path)
    results = {}
    for mode in ("lexical", "hybrid"):
        monkeypatch.setenv("HARNESS_RETRIEVAL_MODE", mode)
        results[mode] = _retrieve(store, enhanced=enhanced, max_passages=max_passages)
    lexical, hybrid = results["lexical"], results["hybrid"]
    assert _ids(lexical) == _ids(hybrid)
    assert lexical["retrieval"]["ranking"]["passage_ids"] == hybrid["retrieval"]["ranking"]["passage_ids"]
    assert lexical["retrieval"]["candidates"] == hybrid["retrieval"]["candidates"]
    assert lexical["retrieval"]["ranking"]["retrieval"] == hybrid["retrieval"]["ranking"]["retrieval"] == "bm25"
    assert lexical["retrieval"]["retrieval_mode"] == "lexical" and hybrid["retrieval"]["retrieval_mode"] == "hybrid"


def test_lexical_mode_ignores_a_dense_index_and_hybrid_mode_fuses_it(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_RETRIEVAL_MODE", "lexical")
    plain = _retrieve(_corpus(tmp_path / "plain"))
    store = _corpus(tmp_path / "dense", dense=True)
    monkeypatch.setattr(store_module, "embed_texts", lambda texts, **_: np.asarray([[1.0, 0.0]], dtype=np.float32))
    assert _ids(_retrieve(store)) == _ids(plain)
    monkeypatch.setenv("HARNESS_RETRIEVAL_MODE", "hybrid")
    fused = _retrieve(store)
    assert fused["retrieval"]["ranking"]["retrieval"] == "bm25+dense"
    assert _ids(fused) != _ids(plain)                         # the dense list is the only difference


def test_fast_reads_the_top_retrieve_k_of_the_ranking(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_RETRIEVAL_MODE", "lexical")
    store = _corpus(tmp_path)
    result = _retrieve(store, retrieve_k=4)
    ranking = result["retrieval"]["ranking"]["passage_ids"]
    assert _ids(result) == ranking[:4]
    assert result["retrieval"]["rank_cutoff"] == 4 and result["retrieval"]["stop_reason"] == "ranked_top_k"
    # No score threshold any more: a passage far below the best BM25 score still fills the cut-off.
    scores = {row["passage_id"]: row["score"]
              for row in store.rank_evidence(QUESTION, passage_ids=_ids(result), limit=10)["items"]}
    assert min(scores.values()) < 0.7 * max(scores.values())


def test_adaptive_budget_chooses_from_a_pool_at_least_as_large_as_itself(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_RETRIEVAL_MODE", "lexical")
    result = _retrieve(_corpus(tmp_path), enhanced=True, retrieve_k=2, max_passages=4)
    assert result["retrieval"]["rank_cutoff"] == 4
    assert len(result["passages"]) == 4


@pytest.mark.parametrize("mode", ["lexical", "hybrid"])
def test_limitations_focus_adds_limitation_passages_from_the_next_ranks(tmp_path: Path, monkeypatch, mode) -> None:
    monkeypatch.setenv("HARNESS_RETRIEVAL_MODE", mode)
    store = _corpus(tmp_path)
    question = "What does MethodX need for sparse attention?"
    general = _retrieve(store, question=question, retrieve_k=2)
    focused = _retrieve(store, question=question, retrieve_k=2,
                        plan=QueryPlan(operation="research", evidence_focus="limitations"))
    # p1-c ranks fourth: inside the next retrieve_k ranks, so it joins; p2-a (third) does not.
    assert _ids(general) == ["p1-b", "p1-a"]
    assert _ids(focused) == ["p1-b", "p1-a", "p1-c"]


@pytest.mark.parametrize("mode", ["lexical", "hybrid"])
def test_several_named_papers_each_get_a_share_of_the_cutoff(tmp_path: Path, monkeypatch, mode) -> None:
    monkeypatch.setenv("HARNESS_RETRIEVAL_MODE", mode)
    store = _corpus(tmp_path)
    question = "How do MethodX Prime Engine and Pruning Study treat sparse attention on long documents?"
    result = _retrieve(store, question=question, retrieve_k=4, paper_ids=["p1", "p3"])
    papers = [str(row["paper_id"]) for row in result["passages"]]
    assert len(papers) == 4 and papers.count("p3") >= 1      # 4 // (2 * 2) = 1 reserved for each


@pytest.mark.parametrize("mode", ["lexical", "hybrid"])
def test_ranks_by_the_full_question_when_the_title_was_all_it_had(tmp_path: Path, monkeypatch, mode) -> None:
    monkeypatch.setenv("HARNESS_RETRIEVAL_MODE", mode)
    result = _retrieve(_corpus(tmp_path), question="What is the MethodX Prime Engine?", paper_ids=["p1"])
    assert result["retrieval"]["ranking"]["query"] == "What is the MethodX Prime Engine?"
    assert set(_ids(result)) <= {"p1-a", "p1-b", "p1-c"} and _ids(result)
