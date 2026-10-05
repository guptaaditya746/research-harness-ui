# corpus

The corpus store and one-time PDF ingest. Papers, passages, DBLP facts and references as JSONL, with BM25 search and optional dense embeddings (hybrid ranking).

## Files

- `store.py` — `CorpusStore` (search, passages, facts, titles; `hybrid_rank`: BM25 + dense fused by RRF, optional rerank)
- `store_tools.py` — the direct-read tools of `CorpusStore` (search, grep, passages, facts, citations, tables, keyphrases)
- `aggregates.py` — `CorpusStore.aggregate`: one function per operation
- `dsl.py`, `rows.py` — the exact search language and bounded regexes; paging and scope helpers
- `dense.py` — `DenseIndex`, `build_dense_index`, `embed_texts`, `rrf` (`make embed` writes `<corpus>/index/dense/`)
- `ingest.py`, `tei.py`, `dblp.py` — GROBID/Docling ingest and DBLP enrichment
- `__main__.py` — `python -m harness_corpus` (see `make ingest`; `--embed` builds the dense index)

## Depends on

harness_config.

See `AGENTS.md` for the repository map and conventions.
