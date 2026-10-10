# Run input: the corpus, or the evidence document D

Every run reads one of two inputs, chosen by `HARNESS_RUN_INPUT` (the Settings page, "What runs read") or per
request (`run_input` on `POST /runs` and `POST /benchmark/execute`, `make benchmark-runs ARGS="--input ..."`):

| Input | What a path does | What its score measures |
|---|---|---|
| `corpus` (default) | Searches the papers, extracts triples, picks and answers: the usual way | Finding + extracting + choosing + answering |
| `evidence_doc` | Gets the question's evidence document D (labelling) and picks the triples that answer, then answers | Choosing + answering, from the same triples for every path |

D is the exhaustive document labelling builds per question (`labelling evidence-docs`; labelling
`docs/EVIDENCE_DOCS.md`). Its reviewed triples are also the gold, so in this mode recall is not limited by what
a path found: the two inputs together separate "did not find it" from "found it but did not choose it".

## Where D comes from

`HARNESS_EVIDENCE_DOC_ROOT` (empty = `$DATASETS_ROOT/evidence_docs`), one folder per question with `doc.json`.
A run finds its document by the question text (case, spacing and punctuation aside). A run with this input and
no document for its question is refused before it starts (HTTP 422). Each triple is given with its first source
that is a corpus passage; triples with none are left out and counted (`run.evidenceDoc.left_out`).

## Per path

- **Fast and Adaptive** (`harness_baseline/given_evidence.py`): no planning, retrieval or extraction. D's
  triples go to the evidence review, which keeps the ones its passage supports and that answer the question
  (`HARNESS_EVIDENCE_DOC_BATCH` triples per call; Fast reviews too, otherwise every triple of D would be its
  answer). Then the usual answer composition (with verification on Adaptive) and answer graph.
- **Agentic** (`harness_agent/doc_research.py`, `doc_tools.py`): its readings are D's interpretations (no model
  call); a reading restricts the run to that interpretation's triples. The supervisor works over D with
  `doc_overview`, `doc_search`, `doc_triples` and `doc_read` only (no corpus tools, no MCP server), records
  findings and names the answering triples. Those go to the same review and answer steps; when it names none,
  all of D's do and the run is marked degraded (`agent_chose_no_triple`).
- **External** (`harness_baseline/external.py`, external-harness `xh run --triples-file`): the agent gets only
  `kbdoc` and `kbcheck` on its PATH and the `instance_doc` prompt; its answer triples are checked as usual.
  Commands that still reached for corpus tools are listed and flagged ("evidence document only" check).
  Its `output.max_triples` (YAML) still caps the answer: raise it for questions over many papers.

## Large documents and the context window

A D of 10,000 triples can leave thousands after review, far more than one prompt holds. Every model call
now fits `HARNESS_MODEL_CONTEXT_TOKENS` (default 32,768, per alias via `HARNESS_MODEL_OPTIONS`
`context_tokens`): the reply never reserves more tokens than the prompt leaves, and a prompt that cannot fit
fails as `context_window_exceeded` (not `provider_error`). The evidence review and the answer split their
input into calls that fit; each answer call is also capped at `HARNESS_ANSWER_BATCH_MAX_TRIPLES` triples.
`HARNESS_MODEL_ANSWER` can select a separate composition model while review and verification keep their
configured models. The answer is composed (and verified) per batch, `HARNESS_ANSWER_BATCH_CONCURRENCY`
at a time, and the batch answers are joined, each sentence with its own source ids. A batch without a
supported answer is left out (the run's "answer composition" check says how many). `HARNESS_ANSWER_MAX_BATCHES`
limits the answer text to N batches spread over the evidence; every reviewed triple stays in the answer graph
and is scored either way. Such runs are long (about one review call per 24 triples, two answer calls per
batch): raise the run deadlines on the Settings page.

## What a run records

`run.runInput` (`corpus` | `evidence_doc`), `run.evidenceDoc` (question id, file, sha256, triples, left out,
interpretation, how many the agent chose) and `run.settings` (every Settings-page value in force). The
retrieval check says "skipped: the input is the evidence document".

## Scoring

The two inputs are never mixed: `make evaluate` scores corpus runs; `make evaluate ARGS="--input evidence_doc"`
scores the others, and the Benchmark page follows the run input in Settings. `--explain` names runs of the
other input.

## The Settings page

Besides the input, the Settings page changes the models, time limits, how much evidence each path reads,
extraction and graph settings and the agent's limits (`harness_config/ui.py` lists them, with their bounds
taken from the settings themselves). Changes apply to runs started afterwards, are refused while a run is
active, are kept in `HARNESS_UI_SETTINGS_FILE` (default `$ARTIFACT_ROOT/ui_settings.json`) and applied over
`.env` and the profile when the API starts. The evidence documents themselves are built in labelling, with its
`HARNESS_DOC_*` settings (how many papers and passages D covers).
