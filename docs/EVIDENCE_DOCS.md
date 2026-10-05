# Evidence documents (D) and a larger, more linked graph

## The idea

For each evaluation question there is one **evidence document D**: everything the corpus says about
the question, written in a fixed vocabulary, built once, reviewed once, and then treated as gold.
Every research path is scored against it. Only two kinds of question are used
(`datasets/question_set_d.json`):

| Category | Questions | Interpretations in D |
|---|---|---|
| `corpus_wide` | nlp32–nlp47 (16) | one: the question as asked |
| `ambiguous` | nlp30, nlp48–nlp53, nlp62–nlp66 (12) | the agent's 3–4 concept readings, each with its own section |

## What a document contains

`datasets/evidence_docs/<question id>/doc.json` (and `doc.md` to read, `<interpretation>.subschema.ttl`),
per interpretation:

| Section | Question it answers |
|---|---|
| `schema` | Which classes and properties may facts use? Seed terms (DBLP, CS-KG core) and invented ones, with origin. |
| `coverage` | Which papers were read, and what was found? One row per indexed paper: `relevant`, `not_relevant`, `no_text`, or `error` (retried on the next build). This is what makes D exhaustive. |
| `triples` | The facts: `s`, `p`, `o`, the paper, and every passage that states it with its evidence. |
| `claims` | Claim subgraphs built from the triples; each points to its triple ids and adds no facts. |
| `answer_key` | Which triples and claims answer this interpretation. Starts as `pending_review`; filled in review. |

Scoring then uses three numbers per run: grounding precision (run triples that are in D), answer
precision and answer recall (against the answer key).

## How it is built

```sh
make evidence-docs                          # every question of HARNESS_DOC_QUESTION_SET
make evidence-docs ARGS="--only nlp33,nlp48"
```

For each interpretation:

1. **Schema**: schema inference over the `HARNESS_DOC_SCHEMA_PASSAGES` highest-ranked passages.
2. **Screening, every paper**: a reader sees the paper's `HARNESS_DOC_SCREEN_PASSAGES` highest-ranked
   passages for the question (or its first passages when none rank) and reports findings or nothing.
3. **Extraction, every passage of a relevant paper** (`HARNESS_DOC_EXTRACT_PASSAGES=0`): open
   extraction, definition and mapping, as in a research run, with the graph settings below.
4. **Claims and files.** The answer key is left for review.

**Watching a build.** In a terminal each stage shows a bar: papers screened (with the number found
relevant) and then passages extracted (with the triples found), the rate, the time left, and the model
calls finished, failed and in flight. Without a terminal (or with `--no-bar`) a line is printed about
every 5%. While a single long call runs (interpretations, schema inference) a line every 20 s says
what it is waiting on. `progress.json` in the question's folder holds the same numbers, so from
another shell:

```sh
make evidence-docs ARGS="--status"                        # one line per question
watch -n 10 cat datasets/evidence_docs/nlp33/progress.json
```

For a quick trial before the full build, `ARGS="--only nlp33 --max-papers 20"` screens only the
first 20 papers (the cache keeps them, so the full build continues from there). The main lever on
time is `HARNESS_DOC_CONCURRENCY` (papers at once), as far as the model server keeps up.

Progress is appended to `cache.jsonl` in the question's folder after every paper, so a stopped build
resumes where it stopped and a finished one is rebuilt without model calls. Delete the folder to
start a question again. Building has no run deadline; `HARNESS_DOC_CALL_TIMEOUT_S` bounds each call
and `HARNESS_DOC_CONCURRENCY` sets how many papers are processed at once. Cost grows with
(questions × interpretations × papers): screening is one reader call per paper, extraction one call
per `HARNESS_EDC_BATCH_PASSAGES` passages of each relevant paper.

## Where triples used to be lost, and the settings for each

| Stage | Before | Setting |
|---|---|---|
| Retrieval | 16–24 passages per run | `BASELINE_RETRIEVE_K`, `HARNESS_*_MAX_PASSAGES`, `HARNESS_RETRIEVAL_MODE` |
| Schema inference | "keep the subschema small" | `HARNESS_SCHEMA_SCOPE=broad`: the question's whole topic |
| Extraction | facts as they come | `HARNESS_GRAPH_LINKING=dense`: one name per entity, entity-to-entity relations, compound statements split into linked facts |
| Relation mapping | an open fact whose relation fits no subschema term was dropped | `HARNESS_UNMAPPED_RELATIONS=keep`: the relation is added (see below); thresholds `HARNESS_MAP_LEXICAL_MIN`, `HARNESS_MAP_MODEL_MIN` |
| Extraction size | 2 passages per call | `HARNESS_EDC_BATCH_PASSAGES` |
| Evidence review | dropped triples were gone | still dropped from the answer, but kept with their verdict in `evidence_graph.json` |

**What "unmapped" means.** Extraction first writes facts in the paper's own words
("MethodX — is fine-tuned from — ModelZ"). Each relation is then matched to a subschema term, by
spelling (`HARNESS_MAP_LEXICAL_MIN`) and then by a mapping model (`HARNESS_MAP_MODEL_MIN`). With
`drop`, a fact whose relation matches nothing is discarded although it is true and sourced. With
`keep`, the relation becomes a new property named from its wording (`isFineTunedFrom`): an OWL 2
sub-property of `researchRelation` (between things) or `researchAttribute` (a value), with domain
Paper when the subject is the paper. It joins the run's subschema, so the graph stays valid and the
scorer's entailed level can relate it to other terms through `predicate_relations.ttl`.

**Answer narrow, evidence wide.** More triples go into the evidence graph, not into the answer:
evidence review still decides what the answer uses (and what gold scoring reads), while
`evidence_graph.json` in every Adaptive and Agentic run keeps every extracted triple with
`relevant: true/false` and the reason, plus the facts lost at mapping.

## Profiles

`profiles/exhaustive.env` sets all of the above in one place (more passages, broad and dense
graphs, unmapped relations kept, one passage per extraction call, long deadlines, every paper and
passage for documents). Use it for any run with `HARNESS_PROFILE=exhaustive` in `.env` or the
environment. Precedence: the environment, then the profile, then `.env`, then the defaults.
`make evidence-docs` uses it unless `HARNESS_PROFILE` names another (`HARNESS_PROFILE=` for none).

## Review

Review each `doc.json` once, with the same people as the gold set (labelling repo): check the
coverage rows of papers marked not relevant (a sample), accept or fix triples, add missing ones,
merge relation names that mean the same (one name stays; the others become rules in
`datasets/ontology/predicate_relations.ttl`), and fill the answer key. Record the reviewed version
with `CORPUS_SNAPSHOT_ID` and `ONTOLOGY_VERSION`; a change to either means rebuilding.
