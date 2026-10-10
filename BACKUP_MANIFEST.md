# Backup manifest

This snapshot preserves the existing 142-commit project history and adds the current source changes. The history scan found no high-confidence API token, cloud key, or private-key markers. The `profiles/exhaustive.env` file in history only contains non-secret runtime settings; current `.env` files are not included.

Excluded local data and generated state:

- The working-tree `datasets/gold/candidates.jsonl` (about 64 MB), backup candidate sets, and `datasets/evidence_docs/` generated document caches.
- Untracked `nlp*/` run folders with progress, cache, and generated evidence files.
- `run_questions.log`, local `.env` files, virtual environments, Node modules, caches, and generated build/results folders.
- Current environment example credential, URL, host, and database fields have been blanked. Original files remain unchanged.

The original private repo and all source files remain on the cluster. Back up excluded corpora and run evidence separately.
