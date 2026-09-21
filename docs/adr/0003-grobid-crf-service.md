# ADR 0003: Pin a CPU GROBID service for M1 ingest

Status: accepted for M1 development; live service verification pending.

Use the official `grobid/grobid:0.9.0-crf` image in `docker/compose.yml`.
GROBID's documented CPU image supports fulltext and reference parsing, which
M1 needs to create citation facts from reference lists. The full image is much
larger and needs a GPU for its best accuracy. The client is pinned through the
Python workspace dependencies. A Docling fallback handles individual rejected
PDFs, though it cannot replace GROBID reference parsing for the M1 citation
exit check.

This compute node cannot access its Docker daemon. The service must be made
reachable before the live M1 exit criterion can run.
