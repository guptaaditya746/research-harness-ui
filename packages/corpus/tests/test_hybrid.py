"""Ranked natural-language retrieval: BM25 without a boolean filter, dense fusion, reranker hook."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from harness_corpus import dense as dense_module
from harness_corpus import store as store_module
from harness_corpus.dense import DenseIndex, passages_digest, rrf
from harness_corpus.store import CorpusStore

PASSAGES = {
    "p1-a": "SafeRAG benchmarks the security of retrieval-augmented generation against knowledge attacks.",
    "p1-b": "Silver noise and inter-context conflict are two of the attack tasks.",
    "p2-a": "HydraRAG unifies graph topology and document semantics for multi-hop reasoning.",
    "p2-b": "Our framework verifies evidence across sources before answering.",
}
TOPIC = {"p1-a": [1, 0, 0], "p1-b": [0.9, 0.1, 0], "p2-a": [0, 1, 0], "p2-b": [0, 0.8, 0.2]}


def _store(tmp_path: Path, *, with_dense: bool) -> CorpusStore:
    papers = [{"paper_id": "p1", "title": "SafeRAG", "year": 2025, "abstract": ""},
              {"paper_id": "p2", "title": "HydraRAG", "year": 2025, "abstract": ""}]
    passages = [{"passage_id": pid, "paper_id": pid.split("-")[0], "section_path": "Body", "para_idx": i,
                 "text": text} for i, (pid, text) in enumerate(PASSAGES.items())]
    for name, rows in (("papers.jsonl", papers), ("passages.jsonl", passages), ("facts.jsonl", []),
                       ("references.jsonl", [])):
        (tmp_path / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    if with_dense:
        folder = tmp_path / "index" / "dense"
        folder.mkdir(parents=True)
        matrix = np.asarray([TOPIC[pid] for pid in PASSAGES], dtype=np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        np.save(folder / "embeddings.npy", matrix.astype(np.float16))
        (folder / "passage_ids.json").write_text(json.dumps(list(PASSAGES)))
        (folder / "meta.json").write_text(json.dumps({"model": "test-embed", "passages_sha256": passages_digest(tmp_path)}))
    store_module._cached_query_embedding.cache_clear()
    return CorpusStore(tmp_path)


def test_natural_language_query_needs_no_boolean_match(tmp_path) -> None:
    store = _store(tmp_path, with_dense=False)
    question = "What security risks does SafeRAG test in retrieval-augmented generation?"
    assert store.search(question)["total"] == 0                 # boolean DSL: every word must appear
    ranked = store.hybrid_rank(question, limit=3)
    assert ranked["retrieval"] == "bm25" and ranked["items"][0]["passage_id"] == "p1-a"
    assert ranked["items"][0]["dense_rank"] is None


def test_dense_ranking_is_fused_with_bm25(tmp_path, monkeypatch) -> None:
    store = _store(tmp_path, with_dense=True)
    monkeypatch.setattr(store_module, "embed_texts",
                        lambda texts, **_: np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32))
    # No word overlap with p2-b, but its embedding is close to the query's topic.
    ranked = store.hybrid_rank("cross-source verification of evidence", limit=4)
    assert ranked["retrieval"] == "bm25+dense"
    ids = [row["passage_id"] for row in ranked["items"]]
    assert set(ids[:2]) == {"p2-a", "p2-b"}
    scoped = store.hybrid_rank("evidence", scope={"paper_ids": ["p1"]}, limit=4)
    assert {row["paper_id"] for row in scoped["items"]} == {"p1"}


def test_embedding_failure_falls_back_to_bm25(tmp_path, monkeypatch) -> None:
    store = _store(tmp_path, with_dense=True)

    def broken(texts, **_):
        import httpx
        raise httpx.ConnectError("embedding server down")

    monkeypatch.setattr(store_module, "embed_texts", broken)
    assert store.hybrid_rank("SafeRAG security", limit=2)["retrieval"] == "bm25"


def test_stale_dense_index_is_ignored(tmp_path) -> None:
    _store(tmp_path, with_dense=True)
    meta = tmp_path / "index" / "dense" / "meta.json"
    meta.write_text(json.dumps({"model": "test-embed", "passages_sha256": "other"}))
    assert DenseIndex(tmp_path).available is False


def test_reranker_reorders_the_fused_head(tmp_path, monkeypatch) -> None:
    store = _store(tmp_path, with_dense=False)
    monkeypatch.setenv("HARNESS_MODEL_RERANK", "test-reranker")

    class Response:
        def raise_for_status(self) -> None: ...

        def json(self) -> dict:
            return {"results": [{"index": 1}, {"index": 0}]}

    monkeypatch.setattr(store_module.httpx, "post", lambda *a, **k: Response())
    ranked = store.hybrid_rank("SafeRAG attack tasks silver noise", limit=2)
    assert ranked["retrieval"].endswith("+rerank")


def test_rrf_and_build(tmp_path, monkeypatch) -> None:
    assert [item for item, _ in rrf([["a", "b"], ["b", "c"]])] == ["b", "a", "c"]
    _store(tmp_path, with_dense=False)
    monkeypatch.setattr(dense_module, "embed_texts",
                        lambda texts, **_: np.asarray([TOPIC[pid] for pid in PASSAGES], dtype=np.float32))
    meta = dense_module.build_dense_index(tmp_path)
    assert meta["count"] == 4 and DenseIndex(tmp_path).available


def test_search_evidence_tool_uses_ranked_retrieval(tmp_path) -> None:
    import asyncio

    from harness_mcp_corpus.server import create_server

    _store(tmp_path, with_dense=False)
    server = create_server(tmp_path)
    result = asyncio.run(server.call_tool("search_evidence", {"query": "Which attack tasks does SafeRAG use?"}))
    payload = json.loads(result[0].text if isinstance(result, list) else result[0][0].text)
    assert payload["items"] and payload["coverage"]["passages_examined"] == 4


@pytest.mark.parametrize("mode", ["lexical", "hybrid"])
def test_retrieval_mode_setting_is_validated(monkeypatch, mode) -> None:
    from harness_config import settings

    monkeypatch.setenv("HARNESS_RETRIEVAL_MODE", mode)
    assert settings().retrieval_mode == mode
