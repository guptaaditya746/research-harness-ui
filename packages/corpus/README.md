# corpus

The corpus store and one-time PDF ingest. Papers, passages, DBLP facts and references as JSONL, with BM25 search and optional dense embeddings (hybrid ranking).

## Files

- `store.py` — `CorpusStore` (search, passages, facts, titles; `hybrid_rank`: BM25, + dense in hybrid mode, fused by RRF, optional rerank)
- `store_tools.py` — the direct-read tools of `CorpusStore` (search, grep, passages, facts, citations, tables, keyphrases)
- `aggregates.py` — `CorpusStore.aggregate`: one function per operation
- `dsl.py`, `rows.py` — the exact search language and bounded regexes; paging and scope helpers
- `dense.py` — `DenseIndex`, `build_dense_index`, `embed_texts`, `rrf` (`make embed` writes `<corpus>/index/dense/`)
- `ingest.py`, `tei.py`, `dblp.py` — GROBID/Docling ingest and DBLP enrichment
- `__main__.py` — `python -m harness_corpus` (see `make ingest`; `--embed` builds the dense index)

## Retrieval

- BM25 (`index/bm25`, built by ingest over passage text): bm25s defaults, Lucene variant, k1 = 1.5,
  b = 0.75, IDF log(1 + (N − df + 0.5) / (df + 0.5)). Tokens (`dsl.bm25_terms`): lowercase
  `\b\w\w+\b`, bm25s' English stopword list, no stemming. A change to any of these needs `make ingest`.
  The External baseline's FTS5 index differs (Porter stemming, k1 = 1.2, FTS5 IDF, own stopwords, the
  section column indexed too), so its scores and matches are not ours.
- `hybrid_rank`: BM25 (+ dense when `HARNESS_RETRIEVAL_MODE=hybrid` and the dense index is current), fused
  by RRF (k = 60, top 1,000 of each list, all given passages when restricted to passage ids; ties by best
  single-list rank, then id). Lexical mode is the same call with the dense list left out. A query
  without BM25 tokens returns nothing. `HARNESS_MODEL_RERANK` reorders the fused top 30 in either mode;
  rows keep the fused `score` and add `rerank_rank` / `rerank_score`.
- `search` (the exact DSL, `dsl.py`): whole lowercase `\w+` words, phrases as consecutive words, so
  every hit on a content word also scores in BM25 ("graph" no longer matches "paragraph").
- Without a BM25 index (small test corpora) both fallbacks count the query's `bm25_terms` in the passage.

## Depends on

harness_config.

See `AGENTS.md` in this folder (entry points, imports, test, rules) and the root `AGENTS.md`.
