# corpus

The corpus store and one-time PDF ingest. Papers, passages, DBLP facts and references as JSONL, with BM25 search.

## Files

- `store.py` — `CorpusStore` (search, passages, facts, titles)
- `ingest.py`, `tei.py`, `dblp.py` — GROBID/Docling ingest and DBLP enrichment
- `__main__.py` — `python -m harness_corpus` (see `make ingest`)

## Depends on

harness_config.

See `AGENTS.md` for the repository map and conventions.
