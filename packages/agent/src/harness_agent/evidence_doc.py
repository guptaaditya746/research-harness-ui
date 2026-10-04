"""Evidence documents (D): one exhaustive document per question, built once and then reviewed.

A document holds everything the corpus says about one question, in a fixed vocabulary, so that every
research path can be scored against it (docs/EVIDENCE_DOCS.md). It has the sections a reviewer
checks, per interpretation of the question:

- ``schema``: the classes and properties facts may use (the seed terms plus invented ones, with origins).
- ``coverage``: one row per indexed paper: relevant, not relevant, or read without result. This is
  what makes the document exhaustive: absence is recorded, not assumed.
- ``triples``: every fact extracted from the relevant papers, with its passages and evidence.
- ``claims``: claim subgraphs built from the triples (they only point to triples, never add facts).
- ``answer_key``: which triples and claims answer the question; filled in during review.

Corpus-wide questions have one interpretation (the question as asked). Ambiguous questions get the
agent's concept readings (3-4), each with its own schema, coverage, triples and answer key.

Building is offline and exhaustive by design: every paper is screened by a reader (its highest-ranked
passages for the question, or its first passages when none rank), and every passage of a relevant paper
goes through open extraction. Progress is appended to ``cache.jsonl`` per question, so an interrupted
build resumes where it stopped. Settings: ``HARNESS_DOC_*`` (.env.example) and the shape of the graph
(``HARNESS_SCHEMA_SCOPE``, ``HARNESS_GRAPH_LINKING``, ``HARNESS_UNMAPPED_RELATIONS``); ``make
evidence-docs`` uses the ``exhaustive`` profile unless another is named.

    make evidence-docs ARGS="--only nlp33,nlp48"
"""

from __future__ import annotations

import argparse
import contextvars
import json
import sys
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from harness_baseline.backbone import schema_terms, subschema_slice
from harness_config import settings
from harness_corpus.store import CorpusStore
from harness_extraction import (
    ExtractedTriple,
    ExtractionEngine,
    OpenAICompatibleModel,
    SchemaInference,
    local_name,
)
from harness_extraction.claims import build_claims
from harness_schema.models import Triple
from harness_semantic import hand_written_subschema
from rdflib import RDFS, URIRef

from .concepts import ConceptProposalError, propose_grounded
from .readers import read_paper

ROOT = Path(__file__).resolve().parents[4]
DOC_VERSION = 1
CLASSES = {"Paper", "Author", "Venue", "Dataset", "Metric", "Claim", "Method", "Task", "Material", "OtherEntity"}


@dataclass
class DocModels:
    """The models a build uses: readers screen papers, the others as in a research run."""

    reader: Any
    schema: Any
    extraction: Any
    edc: Any
    agent: Any

    @classmethod
    def from_settings(cls) -> DocModels:
        config = settings()
        timeout = config.doc_call_timeout_s

        def model(alias: str) -> OpenAICompatibleModel:
            return OpenAICompatibleModel(model=alias, timeout_s=timeout)

        return cls(reader=model(config.model_agent), schema=model(config.model_high),
                   extraction=model(config.model_small), edc=model(config.model_edc or config.model_small),
                   agent=model(config.model_agent))


class Cache:
    """Append-only progress (``key`` -> ``value``, the last line for a key wins); safe across threads."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.values: dict[str, Any] = {}
        self.lock = threading.Lock()
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue                       # a line cut off by an interrupted build
                if isinstance(row, dict) and "key" in row:
                    self.values[row["key"]] = row.get("value")

    def get(self, key: str) -> Any:
        return self.values.get(key)

    def put(self, key: str, value: Any) -> None:
        with self.lock:
            self.values[key] = value
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as handle:
                handle.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")


def research_question(question: str, interpretation: dict[str, Any]) -> str:
    """The question as models see it, with the interpretation attached (as in a research run)."""
    if interpretation.get("source") == "question":
        return question
    return (f"{question}\n\nResearch interpretation chosen by the user: "
            f"{interpretation.get('label', '')}: {interpretation.get('gloss', '')}")


def paper_passages(corpus: CorpusStore, paper_id: str) -> list[dict[str, Any]]:
    """A paper's passages in reading order (the order of the corpus file)."""
    return [row for row in corpus.passages if str(row["paper_id"]) == paper_id]


def screening_rows(corpus: CorpusStore, query: str, paper_id: str, limit: int) -> tuple[list[dict[str, Any]], str]:
    """The passages a screening reader sees: the paper's highest-ranked for the question, or, when none
    ranks, its first passages, so that every paper is actually read."""
    try:
        ranked = corpus.hybrid_rank(query, scope={"paper_ids": [paper_id]}, limit=limit)["items"]
    except Exception:  # noqa: BLE001 - ranking failure falls back to reading order
        ranked = []
    if ranked:
        return ranked, "ranked"
    return paper_passages(corpus, paper_id)[:limit], "first_passages"


def interpretations(row: dict[str, Any], models: DocModels, corpus: CorpusStore, cache: Cache) -> list[dict[str, Any]]:
    """One interpretation for a corpus-wide question; the agent's readings for an ambiguous one."""
    stored = cache.get("interpretations")
    if stored:
        return stored
    if row.get("category") != "ambiguous":
        found = [{"id": "i1", "label": "As asked", "gloss": row["question"], "source": "question"}]
    else:
        try:
            proposal = propose_grounded(models.agent, row["question"], CLASSES, corpus)
        except ConceptProposalError as exc:
            raise RuntimeError(f"{row['id']}: the agent proposed no interpretations ({exc})") from exc
        found = [{"id": f"i{number}", "label": concept.label, "gloss": concept.gloss, "source": "agent",
                  "classes": concept.classes} for number, concept in enumerate(proposal.concepts, start=1)]
    cache.put("interpretations", found)
    return found


def _relation_spec(engine: ExtractionEngine, name: str) -> dict[str, str] | None:
    """How an invented relation is declared, so a resumed build can declare it again."""
    iri = engine.property_names.get(name)
    if iri is None:
        return None
    graph = engine.schema.graph
    domain, range_ = graph.value(iri, RDFS.domain), graph.value(iri, RDFS.range)
    parent = graph.value(iri, RDFS.subPropertyOf)
    return {
        "name": name,
        "definition": str(graph.value(iri, RDFS.comment) or ""),
        "domain": local_name(domain) if isinstance(domain, URIRef) else "Paper",
        "range": ("xsd:string" if isinstance(range_, URIRef) and "XMLSchema" in str(range_)
                  else local_name(range_) if isinstance(range_, URIRef) else "xsd:string"),
        "parent": local_name(parent) if isinstance(parent, URIRef) else "",
    }


class DocumentBuilder:
    """Builds and writes the evidence document of one question (see the module docstring)."""

    def __init__(self, corpus: CorpusStore, models: DocModels, out_root: Path,
                 log: Callable[[str], None] = print) -> None:
        self.corpus = corpus
        self.models = models
        self.out_root = out_root
        self.log = log

    def build(self, row: dict[str, Any]) -> Path:
        folder = self.out_root / str(row["id"])
        cache = Cache(folder / "cache.jsonl")
        readings = interpretations(row, self.models, self.corpus, cache)
        self.log(f"{row['id']}: {len(readings)} interpretation(s)")
        built = [self._interpretation(row, reading, cache, folder) for reading in readings]
        document = {
            "version": DOC_VERSION,
            "question": row,
            "built": self._provenance(),
            "interpretations": built,
            "counts": {
                "interpretations": len(built),
                "papers_screened": max((item["counts"]["papers_screened"] for item in built), default=0),
                "triples": sum(item["counts"]["triples"] for item in built),
                "claims": sum(item["counts"]["claims"] for item in built),
            },
        }
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "doc.json").write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n")
        (folder / "doc.md").write_text(render_markdown(document))
        self.log(f"{row['id']}: {document['counts']['triples']} triples, {document['counts']['claims']} claims "
                 f"-> {folder / 'doc.json'}")
        return folder

    def _papers(self) -> list[str]:
        limit = settings().doc_max_papers
        ids = [str(paper["paper_id"]) for paper in self.corpus.papers]
        return ids[:limit] if limit else ids

    def _interpretation(self, row: dict[str, Any], reading: dict[str, Any], cache: Cache,
                        folder: Path) -> dict[str, Any]:
        config = settings()
        iid = reading["id"]
        question = research_question(row["question"], reading)
        engine = ExtractionEngine(self.models.extraction, hand_written_subschema(),
                                  schema_model=self.models.schema, edc_model=self.models.edc)
        seed_properties = set(engine.property_names)

        # Schema: proposed once, then replayed from the cache.
        stored = cache.get(f"schema:{iid}")
        if stored:
            inferred = engine.apply_schema(SchemaInference.model_validate(stored), question)
        else:
            sample = self.corpus.hybrid_rank(question, limit=config.doc_schema_passages)["items"]
            inferred = engine.infer_schema(question, sample)
            cache.put(f"schema:{iid}", inferred.model_dump())
        self.log(f"{row['id']}/{iid}: schema {len(inferred.classes)} classes, {len(inferred.properties)} properties")

        papers = self._papers()
        concept = None if reading.get("source") == "question" else reading
        coverage = self._screen(row, iid, question, concept, papers, cache)
        relevant = [paper for paper in papers if coverage[paper]["status"] == "relevant"]
        self.log(f"{row['id']}/{iid}: {len(relevant)} of {len(papers)} papers relevant")
        extracted = self._extract(row, iid, question, inferred, engine, relevant, coverage, cache)

        kept = [name for name in engine.kept_relations if name not in inferred.properties]
        used = {triple.predicate for rows in extracted.values() for triple in rows}
        inferred = inferred.model_copy(update={"properties": [
            *inferred.properties, *kept, *sorted(used - set(inferred.properties) - set(kept))]})
        triples = self._merge(extracted)
        claims = self._claims(triples)
        terms = schema_terms(engine.schema.graph, inferred.classes, inferred.properties)
        (folder / f"{iid}.subschema.ttl").parent.mkdir(parents=True, exist_ok=True)
        (folder / f"{iid}.subschema.ttl").write_text(subschema_slice(engine.schema.graph, terms).serialize(format="turtle"))
        for paper in papers:
            coverage[paper]["triples"] = sum(1 for triple in triples if triple["paper_id"] == paper)
        rows = [{"paper_id": paper, **coverage[paper]} for paper in papers]
        return {
            **reading,
            "schema": {"classes": list(inferred.classes), "properties": list(inferred.properties),
                       "terms": [term.model_dump(by_alias=True) for term in terms],
                       "invented": sorted(set(inferred.properties) - seed_properties),
                       "kept_from_extraction": dict(engine.kept_relations)},
            "coverage": rows,
            "triples": triples,
            "claims": claims,
            "answer_key": {"status": "pending_review", "triple_ids": [], "claim_ids": [], "computed": [],
                           "note": "Mark in review which triples and claims answer this interpretation."},
            "counts": {
                "papers_screened": len(papers),
                "relevant": sum(1 for row in rows if row["status"] == "relevant"),
                "relevant_with_triples": sum(1 for row in rows if row["status"] == "relevant" and row["triples"]),
                "not_relevant": sum(1 for row in rows if row["status"] == "not_relevant"),
                "unread": sum(1 for row in rows if row["status"] in {"error", "no_text"}),
                "triples": len(triples),
                "claims": len(claims),
            },
        }

    def _parallel(self, work: Callable[[str], Any], items: list[str]) -> None:
        workers = min(settings().doc_concurrency, max(1, len(items)))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="evidence-doc") as pool:
            for future in [pool.submit(contextvars.copy_context().run, work, item) for item in items]:
                future.result()

    def _screen(self, row: dict[str, Any], iid: str, question: str, concept: dict | None, papers: list[str],
                cache: Cache) -> dict[str, dict[str, Any]]:
        limit = settings().doc_screen_passages
        coverage: dict[str, dict[str, Any]] = {}
        done = [0]

        def screen(paper: str) -> None:
            key = f"screen:{iid}:{paper}"
            stored = cache.get(key)
            if stored is None:
                rows, selection = screening_rows(self.corpus, question, paper, limit)
                title = self.corpus.paper_title(paper)
                if not rows:
                    stored = {"title": title, "status": "no_text", "selection": selection, "passages_read": [],
                              "findings": []}
                else:
                    result = read_paper(self.models.reader, self.corpus, row["question"], concept, paper, limit,
                                        rows=rows)
                    status = "relevant" if result.findings else "error" if result.error else "not_relevant"
                    stored = {"title": title, "status": status, "selection": selection,
                              "passages_read": result.passage_ids, "findings": result.findings,
                              **({"error": result.error} if result.error else {})}
                    if status == "error":
                        # A failed reader is retried on the next build, not recorded as an answer.
                        coverage[paper] = stored
                        return
                cache.put(key, stored)
            coverage[paper] = stored
            done[0] += 1
            if done[0] % 25 == 0:
                self.log(f"{row['id']}/{iid}: screened {done[0]}/{len(papers)}")

        self._parallel(screen, papers)
        return coverage

    def _extract(self, row: dict[str, Any], iid: str, question: str, inferred: SchemaInference,
                 engine: ExtractionEngine, papers: list[str], coverage: dict[str, dict[str, Any]],
                 cache: Cache) -> dict[str, list[ExtractedTriple]]:
        config = settings()
        seed = set(engine.property_names)
        found: dict[str, list[ExtractedTriple]] = {}
        stored_specs: list[dict[str, str]] = []
        for paper in papers:
            stored = cache.get(f"extract:{iid}:{paper}")
            if stored is not None:
                stored_specs.extend(stored.get("relations") or [])
        for spec in stored_specs:
            # Relations invented in an earlier, interrupted build: declare them again before reuse.
            if spec["name"] not in engine.property_names:
                engine.register_local_relation(spec["name"], spec["definition"], domain_name=spec["domain"],
                                               range_name=spec["range"], parent=spec.get("parent") or None)
                engine.kept_relations.setdefault(spec["name"], spec["name"])

        def extract(paper: str) -> None:
            key = f"extract:{iid}:{paper}"
            stored = cache.get(key)
            if stored is None:
                passages = paper_passages(self.corpus, paper)
                if config.doc_extract_passages:
                    wanted = {pid for finding in coverage[paper].get("findings", []) for pid in finding["evidence_ids"]}
                    ranked = screening_rows(self.corpus, question, paper, config.doc_extract_passages)[0]
                    wanted |= {str(item["passage_id"]) for item in ranked}
                    passages = [item for item in passages if str(item["passage_id"]) in wanted]
                size = config.edc_batch_passages
                triples: list[ExtractedTriple] = []
                errors: list[str] = []
                for start in range(0, len(passages), size):
                    try:
                        result, _ = engine.extract_edc(question, inferred, passages[start:start + size])
                        triples.extend(result.triples)
                    except Exception as exc:  # noqa: BLE001 - one failed batch is recorded, not fatal
                        errors.append(f"{type(exc).__name__}: {exc}"[:300])
                if errors and not triples:
                    coverage[paper]["extraction_errors"] = errors
                    return                                     # retried on the next build
                specs = [spec for name in sorted({item.predicate for item in triples} - seed)
                         if (spec := _relation_spec(engine, name))]
                stored = {"triples": [item.model_dump() for item in triples], "relations": specs,
                          "passages": len(passages), "errors": errors}
                cache.put(key, stored)
            found[paper] = [ExtractedTriple.model_validate(item) for item in stored["triples"]]
            if stored.get("errors"):
                coverage[paper]["extraction_errors"] = stored["errors"]

        self._parallel(extract, papers)
        return {paper: found[paper] for paper in papers if paper in found}

    def _merge(self, extracted: dict[str, list[ExtractedTriple]]) -> list[dict[str, Any]]:
        """One row per distinct fact of a paper, with every passage that states it."""
        rows: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        for paper, triples in extracted.items():
            for item in triples:
                key = (paper, item.subject.casefold().strip(), item.predicate, item.object.casefold().strip())
                row = rows.get(key)
                if row is None:
                    rows[key] = {"s": item.subject, "p": item.predicate, "o": item.object, "paper_id": paper,
                                 "sources": [{"source_id": item.source_id, "evidence": item.evidence}]}
                elif all(source["source_id"] != item.source_id for source in row["sources"]):
                    row["sources"].append({"source_id": item.source_id, "evidence": item.evidence})
        ordered = sorted(rows.values(), key=lambda row: (row["paper_id"], row["p"], row["s"].casefold(),
                                                          row["o"].casefold()))
        return [{"id": f"t{number:05d}", **row} for number, row in enumerate(ordered, start=1)]

    def _claims(self, triples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        edges = [Triple(s=row["s"], p=row["p"], o=row["o"], support=f"{len(row['sources'])} passage(s)",
                        evidence=source["evidence"], source=source["source_id"], tripleId=row["id"])
                 for row in triples for source in row["sources"]]
        paper_of = {source["source_id"]: row["paper_id"] for row in triples for source in row["sources"]}
        claims, _ = build_claims(edges, lambda source_id, subject: paper_of.get(source_id, subject))
        by_key = {(row["s"].casefold(), row["p"], row["o"].casefold()): row["id"] for row in triples}
        out = []
        for claim in claims:
            data = claim.model_dump(by_alias=True)
            fields = claim.fields
            key = (fields["subject"].value.casefold(), fields["relation"].value, fields["object"].value.casefold())
            data["tripleIds"] = [by_key[key]] if key in by_key else []
            out.append(data)
        return out

    def _provenance(self) -> dict[str, Any]:
        config = settings()
        return {
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
            "corpus_snapshot_id": config.corpus_snapshot_id,
            "corpus_fingerprint": config.corpus_fingerprint,
            "ontology_version": config.ontology_version,
            "profile": config.profile,
            "papers_in_corpus": len(self.corpus.papers),
            "models": {name: getattr(model, "model", type(model).__name__)
                       for name, model in vars(self.models).items()},
            "settings": {key: getattr(config, key) for key in (
                "schema_scope", "graph_linking", "unmapped_relations", "map_lexical_min", "map_model_min",
                "edc_batch_passages", "content_core", "schema_new_classes", "retrieval_mode", "prompt_set",
                "doc_max_papers", "doc_screen_passages", "doc_extract_passages", "doc_schema_passages")},
        }


def render_markdown(document: dict[str, Any]) -> str:
    """A readable view of the document for reviewers; doc.json stays the source."""
    question = document["question"]
    built = document["built"]
    header = (f"Category: {question.get('category')} · built {built['at']} · "
              f"profile {built.get('profile') or 'none'} · {built['papers_in_corpus']} papers in the corpus")
    lines = [f"# {question['id']}: {question['question']}", "", header, ""]
    for reading in document["interpretations"]:
        counts = reading["counts"]
        summary = (f"Screened {counts['papers_screened']} papers: {counts['relevant']} relevant "
                   f"({counts['relevant_with_triples']} with triples), {counts['not_relevant']} not relevant, "
                   f"{counts['unread']} unread. {counts['triples']} triples, {counts['claims']} claims.")
        lines += [f"## {reading['id']}. {reading['label']}", "", reading["gloss"], "", summary, "",
                  "### Schema", ""]
        for term in reading["schema"]["terms"]:
            lines.append(f"- `{term['name']}` ({term['kind']}, {term['origin']})"
                         + (f": {term['domain']} → {term['range']}" if term.get("domain") else ""))
        lines += ["", "### Triples by paper", ""]
        current = None
        for row in reading["triples"]:
            if row["paper_id"] != current:
                current = row["paper_id"]
                title = next((item["title"] for item in reading["coverage"] if item["paper_id"] == current), None)
                lines += ["", f"**{current}** {title or ''}", "", "| id | subject | relation | object | source |",
                          "|---|---|---|---|---|"]
            sources = ", ".join(source["source_id"] for source in row["sources"])
            lines.append(f"| {row['id']} | {row['s']} | {row['p']} | {row['o']} | {sources} |")
        relevant_empty = [row["paper_id"] for row in reading["coverage"]
                          if row["status"] == "relevant" and not row.get("triples")]
        if relevant_empty:
            lines += ["", "Relevant but nothing extracted: " + ", ".join(relevant_empty)]
        lines += ["", "### Answer key", "", f"Status: {reading['answer_key']['status']}.", ""]
    return "\n".join(lines).rstrip() + "\n"


def load_questions(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text())
    return rows if isinstance(rows, list) else rows.get("questions", [])


def main(argv: list[str] | None = None) -> int:
    config = settings()
    datasets = config.datasets_root or ROOT / "datasets"
    parser = argparse.ArgumentParser(description="Build one exhaustive evidence document per question.")
    parser.add_argument("--questions", type=Path, default=datasets / config.doc_question_set)
    parser.add_argument("--only", default="", help="comma-separated question ids")
    parser.add_argument("--out", type=Path, default=config.doc_root or datasets / "evidence_docs")
    args = parser.parse_args(argv)
    questions = load_questions(args.questions)
    if args.only:
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
        questions = [row for row in questions if row.get("id") in wanted]
    corpus = CorpusStore(config.corpus_root or ROOT / "corpus_data")
    builder = DocumentBuilder(corpus, DocModels.from_settings(), args.out,
                              log=lambda text: print(text, flush=True))
    failed = []
    for row in questions:
        try:
            builder.build(row)
        except Exception as exc:  # noqa: BLE001 - one question's failure must not stop the others
            failed.append(row.get("id"))
            print(f"{row.get('id')}: failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    print(f"{len(questions) - len(failed)} documents built" + (f"; failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["Cache", "DocModels", "DocumentBuilder", "interpretations", "render_markdown", "screening_rows"]
