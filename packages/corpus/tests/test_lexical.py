"""The exact search language matches whole words, consistent with the BM25 tokenizer."""

from __future__ import annotations

import json
from pathlib import Path

import bm25s
import pytest
from harness_corpus.dsl import bm25_terms, matches
from harness_corpus.store import CorpusStore

PASSAGES = {
    "p1-a": "Knowledge graph embeddings of MethodX are trained with a margin loss.",
    "p1-b": "We embed graphs with a neural encoder; each paragraph is a node.",
    "p2-a": "A knowledge graph embedding model scores triples on BenchY.",
    "p2-b": "Graph neural networks propagate messages between nodes.",
    "p3-a": "Pasta recipes rarely mention graphs; we use GPT-4 and not GPT-3.",
}


def _store(tmp_path: Path, *, indexed: bool) -> CorpusStore:
    papers = [{"paper_id": f"p{i}", "title": f"Paper {i}", "year": 2024, "abstract": ""} for i in (1, 2, 3)]
    passages = [{"passage_id": pid, "paper_id": pid.split("-")[0], "section_path": "Body", "para_idx": i,
                 "text": text} for i, (pid, text) in enumerate(PASSAGES.items())]
    for name, rows in (("papers.jsonl", papers), ("passages.jsonl", passages), ("facts.jsonl", []),
                       ("references.jsonl", [])):
        (tmp_path / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    if indexed:
        folder = tmp_path / "index" / "bm25"
        folder.mkdir(parents=True)
        retriever = bm25s.BM25()
        retriever.index(bm25s.tokenize([p["text"] for p in passages], stopwords="en", show_progress=False),
                        show_progress=False)
        retriever.save(str(folder), corpus=[{"passage_id": p["passage_id"]} for p in passages],
                       show_progress=False)
    return CorpusStore(tmp_path)


def test_terms_match_whole_words_not_substrings() -> None:
    assert not matches("graph", "each paragraph is a node")      # was a substring hit
    assert matches("graph", "a Graph, here")
    assert not matches("graph", "we embed graphs")               # no stemming, like BM25
    assert matches("graph NOT paragraph", "graph only")
    assert not matches("graph", "graphical models")


def test_phrases_are_consecutive_words() -> None:
    assert matches('"knowledge graph"', "A knowledge-graph model")
    assert not matches('"knowledge graph"', "graph knowledge")
    assert not matches('"edge graph"', "knowledge graph")        # was a substring hit across words
    assert matches('"GPT 4"', "we use GPT-4 today")
    assert not matches('"GPT 4"', "GPT-3 and 4 others")
    assert matches("model NEAR/1 triples", "model scores triples")
    assert not matches("model NEAR/0 triples", "model scores triples")


@pytest.mark.parametrize("text", [
    "The GPT-4 model, state-of-the-art graphs!", "what is the", "", "Größe ÉTÉ naïve x_y 2024 a1 I",
    "MethodX's results on BenchY (F1 = 0.93)",
])
def test_bm25_terms_are_the_bm25s_tokens(text: str) -> None:
    tokenized = bm25s.tokenize([text], stopwords="en", show_progress=False)
    inverse = {index: token for token, index in tokenized.vocab.items()}
    assert bm25_terms(text) == [inverse[index] for index in tokenized.ids[0]]


def test_every_exact_hit_has_a_positive_bm25_score(tmp_path: Path) -> None:
    store = _store(tmp_path, indexed=True)
    for query in ("graph", "graph OR embedding", '"knowledge graph"', "graphs"):
        hits = [row["passage_id"] for row in store.search(query, limit=10)["items"]]
        assert hits, query
        scores = store.rank_evidence(query.replace('"', "").replace(" OR ", " "), passage_ids=hits, limit=10)
        assert all(row["score"] > 0 for row in scores["items"]), query
    assert {row["passage_id"] for row in store.search("graph", limit=10)["items"]} == {"p1-a", "p2-a", "p2-b"}


def test_no_index_fallbacks_share_the_bm25_tokens(tmp_path: Path) -> None:
    store = _store(tmp_path, indexed=False)
    # Three-letter words count in both fallbacks now (they used > 2 and > 3 characters); whole words only.
    evidence = {row["passage_id"]: row["score"] for row in store.rank_evidence("use graph", limit=10)["items"]}
    ranked = dict(store._bm25_order("use graph", set(PASSAGES)))
    assert ranked == {pid: score for pid, score in evidence.items() if score > 0}
    assert set(ranked) == {"p1-a", "p2-a", "p2-b", "p3-a"}      # p3-a by "use"; "paragraph" is not "graph"
