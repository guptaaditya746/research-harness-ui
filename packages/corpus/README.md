# corpus

The corpus store and one-time PDF ingest. Papers, passages, DBLP facts and references as JSONL, with BM25 search and optional dense embeddings (hybrid ranking).

## Files

- `store.py` — `CorpusStore` (search, passages, facts, titles; `hybrid_rank`: BM25 + dense fused by RRF, optional rerank)
- `dense.py` — `DenseIndex`, `build_dense_index`, `embed_texts`, `rrf` (`make embed` writes `<corpus>/index/dense/`)
- `ingest.py`, `tei.py`, `dblp.py` — GROBID/Docling ingest and DBLP enrichment
- `__main__.py` — `python -m harness_corpus` (see `make ingest`; `--embed` builds the dense index)

## Depends on

harness_config.

See `AGENTS.md` for the repository map and conventions.
