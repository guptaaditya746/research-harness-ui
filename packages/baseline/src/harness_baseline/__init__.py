"""Baseline-20 as a fixed four-node LangGraph workflow."""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from difflib import SequenceMatcher
from typing import Any, TypedDict
from urllib.parse import quote

from harness_corpus.store import CorpusStore
from harness_extraction import (
    AnswerResult,
    ExtractedTriple,
    ExtractionEngine,
    ModelFailure,
    SchemaInference,
)
from harness_extraction.claims import (
    CHANGE,
    NUMERIC,
    build_claims,
    classify_claim_relations,
    derive_query_claim,
    enrich_result_claims,
    valid_edge,
)
from harness_graphstore import ProvenanceDataset, sparql_check
from harness_observability import record_stage, record_tool_call
from harness_schema.models import (
    ClaimSubgraph,
    Metrics,
    Reading,
    Run,
    TraceRow,
    Triple,
    required_artifacts,
)
from harness_semantic import DBLP, HRN, validate_subschema
from langgraph.graph import END, START, StateGraph
from rdflib import RDFS, XSD, Literal, URIRef

TraceSink = Callable[[TraceRow], None]


class BaselineState(TypedDict, total=False):
    question: str
    passages: list[dict[str, Any]]
    inferred: SchemaInference
    extracted: list[ExtractedTriple]
    answer: AnswerResult
    retrieval: dict[str, Any]
    extraction_batches: list[dict[str, Any]]


class BaselineResult(TypedDict):
    run: Run
    artifacts: dict[str, bytes]


class NoResults(RuntimeError):
    pass


class CorpusAbsent(NoResults):
    def __init__(self, message: str, evidence_surface: str, inspected: int):
        super().__init__(message)
        self.evidence_surface = evidence_surface
        self.inspected = inspected


class NoEvidenceFound(NoResults):
    """A completed semantic search, deliberately weaker than corpus absence."""
    def __init__(self, message: str, coverage: dict[str, Any]):
        super().__init__(message)
        self.coverage = coverage


def retrieval_dsl(question: str) -> str:
    stop = {
        "about",
        "after",
        "before",
        "could",
        "does",
        "from",
        "have",
        "paper",
        "papers",
        "that",
        "their",
        "these",
        "this",
        "what",
        "when",
        "where",
        "which",
        "with",
        "would",
    }
    terms = []
    for term in re.findall(r"[A-Za-z][A-Za-z0-9_-]+", question):
        if (
            len(term) >= 4
            and term.casefold() not in stop
            and term.casefold() not in {t.casefold() for t in terms}
        ):
            terms.append(term)
    if not terms:
        raise NoResults("question has no searchable terms")
    return " OR ".join(terms[:12])


class Baseline20:
    def __init__(
        self,
        corpus: CorpusStore,
        extraction: ExtractionEngine,
        retrieve_k: int = 20,
        enhanced: bool = False,
        edc: bool = False,
        trace_sink: TraceSink | None = None,
        refine: bool = False,
        refinement_model: Any | None = None,
        claim_lookup: Callable[[str], list[ClaimSubgraph]] | None = None,
        relation_proposer: Callable[[str, str, str | None], dict] | None = None,
        seed_passage_ids: list[str] | None = None,
    ) -> None:
        self.corpus = corpus
        self.extraction = extraction
        self.retrieve_k = retrieve_k
        self.enhanced = enhanced
        self.edc = edc
        self.trace_sink = trace_sink or (lambda _: None)
        self.refine = refine
        self.refinement_model = refinement_model
        self.claim_lookup = claim_lookup
        self.relation_proposer = relation_proposer
        self.seed_passage_ids = seed_passage_ids or []
        self.started = 0.0
        self.trace: list[TraceRow] = []
        self.current_run_id: str | None = None
        self.graph = self._graph()

    def _emit(self, action: str, returned: str) -> None:
        row = TraceRow(
            atS=round(time.monotonic() - self.started, 3), action=action, returned=returned
        )
        self.trace.append(row)
        self.trace_sink(row)
        record_stage(action, returned, run_id=self.current_run_id)

    def _graph(self):
        graph = StateGraph(BaselineState)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("infer_schema", self._infer_schema)
        graph.add_node("extract_build", self._extract_build)
        graph.add_node("query_answer", self._query_answer)
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "infer_schema")
        graph.add_edge("infer_schema", "extract_build")
        graph.add_edge("extract_build", "query_answer")
        graph.add_edge("query_answer", END)
        return graph.compile()

    def _retrieve(self, state: BaselineState) -> dict[str, Any]:
        question = state["question"]
        mentioned_papers = self.corpus.mentioned_paper_ids(question)
        dsl = retrieval_dsl(question)
        queries = [dsl]
        if self.enhanced:
            terms = [t for t in re.findall(r"[A-Za-z][A-Za-z0-9_-]+", question) if len(t) >= 4]
            phrases = [f'"{" ".join(terms[i : i + 2])}"' for i in range(max(0, len(terms) - 1))]
            queries.extend(phrases[:4])
        found: dict[str, dict[str, Any]] = {}
        rounds: list[dict[str, Any]] = []
        for round_no, query in enumerate(queries, start=1):
            # Scope explicit paper questions before broad lexical expansion.
            # Inspect all matches, then use the existing BM25S index to select
            # evidence by relevance rather than by a fixed passage count.
            per_query_limit = max(1, len(self.corpus.passages))
            query_started = time.monotonic()
            scope = {"paper_ids": mentioned_papers} if mentioned_papers else None
            hits = self.corpus.search(query, scope=scope, limit=per_query_limit)["items"]
            before = len(found)
            found.update({row["passage_id"]: row for row in hits})
            rounds.append({
                "round": round_no, "query": query, "limit": per_query_limit,
                "returned": len(hits), "new": len(found) - before,
                "passage_ids": [str(row["passage_id"]) for row in hits],
                "duration_ms": round((time.monotonic() - query_started) * 1000, 1),
            })
        reused_claims = self.claim_lookup(question) if self.claim_lookup else []
        for claim in reused_claims:
            for source_id in claim.source_ids:
                passage = self.corpus.by_passage.get(source_id)
                if passage is not None:
                    found.setdefault(source_id, passage)
        for source_id in self.seed_passage_ids:
            passage = self.corpus.by_passage.get(source_id)
            if passage is not None:
                found.setdefault(source_id, passage)
        if not found:
            raise NoEvidenceFound("No evidence was found in the indexed full text.", {
                "coverage_complete": False, "queries": len(queries), "passages_inspected": 0,
                "papers_in_scope": len(mentioned_papers) or len(self.corpus.papers),
                "retrieval_methods": "lexical+semantic-rank", "parser_failures": 0,
            })
        focus = question
        for paper_id in mentioned_papers:
            title = self.corpus.paper_title(paper_id) or ""
            prefix = " ".join(title.split()[:3])
            if prefix:
                focus = re.sub(re.escape(prefix), " ", focus, flags=re.IGNORECASE)
            brand = re.split(r"[^A-Za-z0-9]+", title, maxsplit=1)[0]
            if brand and (brand.isupper() or any(char.isdigit() for char in brand) or any(char.isupper() for char in brand[1:])):
                focus = re.sub(rf"(?i)\b{re.escape(brand)}\b", " ", focus)
        if not re.search(r"[A-Za-z]{3,}", focus):
            focus = question
        rank_started = time.monotonic()
        ranked = self.corpus.rank_evidence(
            focus, passage_ids=list(found), limit=max(1, len(found))
        )["items"]
        rank_duration_ms = round((time.monotonic() - rank_started) * 1000, 1)
        best_by_paper: dict[str, float] = {}
        for row in ranked:
            key = str(row["paper_id"]) if mentioned_papers else "all"
            best_by_paper[key] = max(best_by_paper.get(key, 0.0), float(row["score"]))
        relative_threshold = 0.70
        limitation_question = "limitation" in question.casefold()
        rows = [
            {key: value for key, value in row.items() if key != "score"}
            for row in ranked
            if row["score"] > 0
            and (
                row["score"] >= relative_threshold * best_by_paper[
                    str(row["paper_id"]) if mentioned_papers else "all"
                ]
                or limitation_question and re.search(r"\blimitations?\b", row["text"], re.IGNORECASE)
            )
        ]
        if not self.enhanced:
            rows = rows[: self.retrieve_k]
        if not rows:
            raise NoEvidenceFound("No evidence was found after the configured full-text search.", {
                "coverage_complete": False, "queries": len(queries), "passages_inspected": len(ranked),
                "papers_in_scope": len(mentioned_papers) or len(self.corpus.papers),
                "retrieval_methods": "lexical+semantic-rank", "parser_failures": 0,
            })
        self._emit("Retrieving top passages", f"{len(rows)} passages across {len(rounds)} rounds")
        return {"passages": rows, "retrieval": {
            "queries": rounds, "stop_reason": "relative_bm25_relevance",
            "mentioned_paper_ids": mentioned_papers,
            "reused_claim_ids": [claim.id for claim in reused_claims],
            "agent_seed_passage_ids": self.seed_passage_ids,
            "candidates": len(found), "selected": len(rows),
            "relative_score_threshold": relative_threshold,
            "ranking": {
                "query": focus, "candidate_ids": list(found),
                "passage_ids": [str(row["passage_id"]) for row in ranked],
                "duration_ms": rank_duration_ms,
            },
        }}

    def _infer_schema(self, state: BaselineState) -> dict[str, Any]:
        inferred = self.extraction.infer_schema(state["question"], state["passages"])
        self._emit(
            "Inferring schema",
            f"{len(inferred.classes)} classes · {len(inferred.properties)} properties",
        )
        return {"inferred": inferred}

    def _extract_build(self, state: BaselineState) -> dict[str, Any]:
        if self.edc:
            extracted = []
            reports = []
            # Small evidence batches keep the structured response focused and
            # avoid a single long generation blocking the whole run.
            batch_size = 2
            for start in range(0, len(state["passages"]), max(1, batch_size)):
                extraction, batch_reports = self.extraction.extract_edc(
                    state["question"], state["inferred"], state["passages"][start : start + batch_size]
                )
                extracted.extend(extraction.triples)
                reports.append({"batch": start // max(1, batch_size) + 1, "passages": len(state["passages"][start : start + batch_size]), "edc": batch_reports})
        elif self.enhanced:
            extracted, reports = self.extraction.extract_batches(
                state["question"], state["inferred"], state["passages"]
            )
        else:
            extracted = self.extraction.extract(
                state["question"], state["inferred"], state["passages"]
            ).triples
            reports = []
        if not extracted:
            # Broad comparison questions often make the model return prose
            # rather than ontology-shaped triples. Recover verified metadata
            # facts so the fixed baseline can still answer with provenance.
            allowed = set(state["inferred"].properties)
            fallback: list[ExtractedTriple] = []
            for row in self.corpus.seed_facts():
                predicate = {"author": "authoredBy"}.get(row["predicate"], row["predicate"])
                if predicate not in allowed:
                    continue
                fallback.append(
                    ExtractedTriple(
                        subject=str(row["subject"]),
                        predicate=predicate,
                        object=str(row["object"]),
                        source_id=str(row["source_id"]),
                        evidence=f"Corpus fact {row.get('fact_id', row['source_id'])}.",
                    )
                )
                if not self.enhanced and len(fallback) >= self.retrieve_k:
                    break
            extracted = fallback
            reports.append({"fallback": "corpus-seed", "triples": len(extracted)})
        if not extracted:
            raise NoResults("extractor and corpus seed produced no supported triples")
        extracted = [
            item for item in extracted
            if all(str(value).strip().casefold() not in {"", "null", "none", "n/a", "undefined"}
                   for value in (item.subject, item.predicate, item.object, item.source_id, item.evidence))
        ]
        normalized: list[ExtractedTriple] = []
        added_relations: set[str] = set()
        for item in extracted:
            if item.predicate == "usesDataset":
                object_text = item.object.casefold()
                evidence_text = item.evidence.casefold()
                replacement = None
                if re.search(r"\b(model|representation)\b", object_text):
                    replacement = ("usesModel", "A paper reports use of a model or representation.")
                elif "environment" in object_text or re.search(r"\benvironments?\b", evidence_text):
                    replacement = ("evaluatedInEnvironment", "A paper reports evaluation in an environment.")
                if replacement:
                    name, definition = replacement
                    self.extraction.register_local_relation(name, definition)
                    if self.relation_proposer:
                        self.relation_proposer(name, definition, self.current_run_id)
                    added_relations.add(name)
                    item = item.model_copy(update={"predicate": name})
            normalized.append(item)
        extracted = normalized
        if added_relations:
            state["inferred"] = state["inferred"].model_copy(update={
                "properties": list(dict.fromkeys([*state["inferred"].properties, *sorted(added_relations)])),
            })
            reports.append({"stage": "local_subschema", "relations": sorted(added_relations)})
        if not extracted:
            raise NoResults("extraction contained no valid supported propositions")
        if self.refine:
            extracted, review = self.extraction.refine_triples(
                state["question"], extracted, state["passages"], reviewer=self.refinement_model
            )
            reports.append({"stage": "evidence_review", **review})
            if not extracted:
                raise NoResults("No in-corpus evidence found for a question-relevant triple")
        mentioned = self.corpus.mentioned_paper_ids(state["question"])
        if self.enhanced and len(mentioned) > 1 and re.search(r"\bmetrics?\b", state["question"], re.IGNORECASE):
            covered = {
                self.corpus.by_passage[item.source_id]["paper_id"]
                for item in extracted if item.source_id in self.corpus.by_passage
            }
            recovered = 0
            for paper_id in mentioned:
                if paper_id in covered:
                    continue
                for passage in state["passages"]:
                    if passage["paper_id"] != paper_id:
                        continue
                    for sentence in re.split(r"(?<=[.!?])\s+", passage["text"]):
                        match = re.search(
                            r"\b(?:reports?|measures?)\s+(?:both\s+)?(.+?)(?:,\s|\.|$)",
                            sentence, re.IGNORECASE,
                        )
                        if not match or not re.search(
                            r"\b(metric|quality|accuracy|PSNR|SSIM|rate|score)\b",
                            match.group(1), re.IGNORECASE,
                        ):
                            continue
                        for criterion in re.split(r"\s+and\s+", match.group(1)):
                            criterion = criterion.strip(" ,;.")
                            if not criterion:
                                continue
                            extracted.append(ExtractedTriple(
                                subject=paper_id, predicate="reportsMetric", object=criterion,
                                source_id=str(passage["passage_id"]), evidence=sentence,
                            ))
                            recovered += 1
                    if recovered:
                        break
            if recovered:
                reports.append({"stage": "missing_paper_metric_coverage", "triples": recovered})
        self._emit("Extracting triples", f"{len(extracted)} supported triples")
        return {"extracted": extracted, "inferred": state["inferred"], "extraction_batches": reports}

    def _query_answer(self, state: BaselineState) -> dict[str, Any]:
        direct, direct_triples = self._direct_metadata_answer(state)
        if direct is not None:
            self._emit("Querying graph and composing answer", f"{len(direct.cited_source_ids)} cited passages")
            return {"answer": direct, "extracted": direct_triples}
        if re.search(r"\b(improv\w*|increase|decrease|reduction|gain|comparison)\b", state["question"], re.IGNORECASE):
            target_context = re.search(r"\bon\s+(.+?)\??$", state["question"], re.IGNORECASE)
            context_terms = (
                {term.casefold() for term in re.findall(r"[A-Za-z]{4,}", target_context.group(1))}
                if target_context else set()
            )
            numeric = [
                item for item in state["extracted"]
                if item.predicate == "reportsMetric" and NUMERIC.search(item.object)
                and CHANGE.search(f"{item.object} {item.evidence}")
                and (not context_terms or any(term in item.object.casefold() for term in context_terms))
            ]
            if numeric:
                selected = list({(item.subject, item.object, item.source_id): item for item in numeric}.values())
                answer = AnswerResult(
                    sparql="SELECT ?paper ?result WHERE { ?paper <https://harness.local/schema#reportsMetric> ?result . }",
                    answer="The indexed evidence reports: " + "; ".join(
                        f"{item.object} [{item.source_id}]" for item in selected
                    ) + ".",
                    cited_source_ids=sorted({item.source_id for item in selected}),
                )
                return {"answer": answer, "extracted": selected}
        answer = self.extraction.answer(
            state["question"], state["inferred"], state["extracted"],
            verify=self.refine, verifier=self.refinement_model,
        )
        try:
            checked_query = sparql_check(answer.sparql, self.extraction.schema)
        except ValueError:
            # The answer and its citations have already passed evidence review.
            # A malformed model query must not discard those verified results.
            predicates = [
                self.extraction.property_names[item.predicate]
                for item in state["extracted"]
                if item.predicate in self.extraction.property_names
            ]
            if not predicates:
                raise
            checked_query = sparql_check(
                f"SELECT ?subject ?object WHERE {{ ?subject <{predicates[0]}> ?object . }}",
                self.extraction.schema,
            )
            self._emit("Repairing graph query", f"Using verified predicate {state['extracted'][0].predicate}")
        answer = answer.model_copy(update={"sparql": checked_query})
        self._emit(
            "Querying graph and composing answer", f"{len(answer.cited_source_ids)} cited passages"
        )
        return {"answer": answer}

    def _direct_metadata_answer(
        self, state: BaselineState
    ) -> tuple[AnswerResult | None, list[ExtractedTriple]]:
        """Answer metadata questions from verified corpus facts.

        The model often infers a broad comparison schema for simple metadata
        questions. Routing these questions through verified facts keeps the
        answer and emitted triples aligned with the user's actual question.
        """

        structural = self._direct_structural_answer(state["question"])
        if structural is not None:
            return structural

        question = state["question"].casefold()
        if "who authored" in question or "authors of" in question or "who wrote" in question:
            target = self._find_title_fact(question)
            if target is not None:
                triples = self._author_triples(target["subject"])
                names = [item.object for item in triples]
                return (
                    AnswerResult(
                        sparql="PREFIX dblp: <https://dblp.org/rdf/schema#>\nSELECT ?author WHERE { ?paper dblp:authoredBy ?author . }",
                        answer=f"{target['object']} was authored by {', '.join(names)} [{target['source_id']}].",
                        cited_source_ids=sorted({item.source_id for item in triples}),
                    ),
                    triples,
                )

        citation_lookup = bool(re.search(r"\b(?:which|what) papers\b.*\bcite\b|\bwho cites\b", question))
        if citation_lookup:
            target = self._find_title_fact(question)
            if target is not None:
                rows = self.corpus.citations(target["subject"], "in", 0, 50)["items"]
                triples = [
                    ExtractedTriple(
                        subject=str(row["paper_id"]),
                        predicate="cites",
                        object=str(target["subject"]),
                        source_id=str(row["reference_id"]),
                        evidence=str(row.get("title") or row.get("raw") or row["fact_id"]),
                    )
                    for row in rows
                ]
                if triples:
                    names = [self.corpus.paper_title(item.subject) or item.subject for item in triples]
                    answer = f"{', '.join(names)} cite {target['object']} [{', '.join(item.source_id for item in triples)}]."
                else:
                    raise CorpusAbsent(
                        f"No in-corpus evidence found that a paper cites {target['object']}. "
                        "This does not establish that no external citation exists.",
                        "citation_index", len(self.corpus.references),
                    )
                return (
                    AnswerResult(
                        sparql="PREFIX dblp: <https://dblp.org/rdf/schema#>\nSELECT ?paper WHERE { ?paper dblp:cites ?target . }",
                        answer=answer,
                        cited_source_ids=sorted({item.source_id for item in triples}),
                    ),
                    triples,
                )

        if "more than one paper" in question and "author" in question:
            grouped: dict[str, list[dict[str, Any]]] = {}
            for row in self.corpus.seed_facts():
                if row.get("predicate") == "author":
                    grouped.setdefault(str(row["object"]), []).append(row)
            repeated = {
                name: works
                for name, rows in grouped.items()
                if len(works := self._distinct_works(rows)) > 1
            }
            triples = [
                ExtractedTriple(
                    subject=str(row["subject"]), predicate="authoredBy", object=name,
                    source_id=str(row["source_id"]), evidence=f"Corpus fact {row.get('fact_id', row['source_id'])}.",
                )
                for name, rows in sorted(repeated.items())
                for row in rows
            ]
            lines = [
                f"{name} ({len(rows)} papers): "
                + ", ".join(self.corpus.paper_title(r["subject"]) or r["subject"] for r in rows)
                for name, rows in sorted(repeated.items())
            ]
            return (
                AnswerResult(
                    sparql="PREFIX dblp: <https://dblp.org/rdf/schema#>\nSELECT ?paper ?author WHERE { ?paper dblp:authoredBy ?author . }",
                    answer="Authors appearing on more than one paper:\n" + ("\n".join(lines) or "None."),
                    cited_source_ids=sorted({item.source_id for item in triples}),
                ),
                triples,
            )

        if ("which paper" in question or "what paper" in question) and not citation_lookup:
            target = self._find_title_fact(question)
            if target is not None:
                triple = ExtractedTriple(
                    subject=str(target["subject"]), predicate="title", object=str(target["object"]),
                    source_id=str(target["source_id"]),
                    evidence=f"Corpus fact {target.get('fact_id', target['source_id'])}.",
                )
                canonical = self.corpus.canonical_paper_iri(str(target["subject"]))
                return (
                    AnswerResult(
                        sparql="PREFIX dblp: <https://dblp.org/rdf/schema#>\nSELECT ?paper ?title WHERE { ?paper dblp:title ?title . }",
                        answer=f"{target['object']} ({canonical or target['subject']}) [{target['source_id']}].",
                        cited_source_ids=[str(target["source_id"])],
                    ),
                    [triple],
                )
        return (None, [])

    @staticmethod
    def _fact_triple(row: dict[str, Any]) -> ExtractedTriple:
        return ExtractedTriple(
            subject=str(row["subject"]),
            predicate={"author": "authoredBy"}.get(str(row["predicate"]), str(row["predicate"])),
            object=str(row["object"]),
            source_id=str(row["source_id"]),
            evidence=f"Corpus fact {row.get('fact_id', row['source_id'])}.",
        )

    @staticmethod
    def _citation_triple(row: dict[str, Any]) -> ExtractedTriple:
        return ExtractedTriple(
            subject=str(row["paper_id"]), predicate="cites",
            object=str(row["resolved_paper_id"]),
            source_id=str(row["reference_id"]),
            evidence=str(row.get("title") or row.get("raw") or row["reference_id"]),
        )

    @staticmethod
    def _fact_answer(
        triples: list[ExtractedTriple], answer: str
    ) -> tuple[AnswerResult, list[ExtractedTriple]]:
        predicate = triples[0].predicate
        return (
            AnswerResult(
                sparql=(
                    "PREFIX hrn: <https://harness.local/schema#>\n"
                    f"SELECT ?subject ?object WHERE {{ ?subject hrn:{predicate} ?object . }}"
                ),
                answer=answer,
                cited_source_ids=sorted({item.source_id for item in triples}),
            ),
            triples,
        )

    def _direct_structural_answer(
        self, raw_question: str
    ) -> tuple[AnswerResult, list[ExtractedTriple]] | None:
        """Answer structural corpus questions from indexed facts and resolved references."""
        question = raw_question.casefold()
        facts = self.corpus.seed_facts()

        if "in what year" in question or "publication year" in question and "earliest" not in question and "latest" not in question:
            target = self._find_title_fact(question)
            if target:
                rows = [row for row in facts if row["subject"] == target["subject"] and row["predicate"] == "year"]
                if rows:
                    triples = [self._fact_triple(row) for row in rows]
                    return self._fact_answer(triples, f"{target['object']} was published in {rows[0]['object']} [{rows[0]['source_id']}].")

        if "in which venue" in question or "where was" in question and "published" in question:
            target = self._find_title_fact(question)
            if target:
                rows = [row for row in facts if row["subject"] == target["subject"] and row["predicate"] == "venue"]
                if rows:
                    triples = [self._fact_triple(row) for row in rows]
                    return self._fact_answer(triples, f"{target['object']} was published in {rows[0]['object']} [{rows[0]['source_id']}].")

        if "which indexed papers does" in question and " cite" in question:
            subject_text = question.split("does", 1)[1].rsplit(" cite", 1)[0]
            source = self._find_title_fact(subject_text)
            if source:
                rows = [row for row in self.corpus.references if row["paper_id"] == source["subject"] and row.get("resolved_paper_id")]
                if not rows:
                    raise CorpusAbsent(f"No resolved in-corpus citations found for {source['object']}.", "citation_index", len(self.corpus.references))
                triples = [self._citation_triple(row) for row in rows]
                names = [self.corpus.paper_title(item.object) or item.object for item in triples]
                return self._fact_answer(triples, f"{source['object']} cites these indexed papers: {', '.join(names)} [{', '.join(item.source_id for item in triples)}].")

        if "how many indexed papers cite" in question:
            target = self._find_title_fact(question)
            if target:
                rows = [row for row in self.corpus.references if row.get("resolved_paper_id") == target["subject"]]
                if not rows:
                    raise CorpusAbsent(f"No resolved in-corpus citations found for {target['object']}.", "citation_index", len(self.corpus.references))
                triples = [self._citation_triple(row) for row in rows]
                count = len({item.subject for item in triples})
                return self._fact_answer(triples, f"{count} indexed papers cite {target['object']} [{', '.join(item.source_id for item in triples)}].")

        citation_pair = re.match(r"does (.+?) cite (.+?)\??$", question)
        if citation_pair:
            source = self._find_title_fact(citation_pair.group(1))
            target = self._find_title_fact(citation_pair.group(2))
            if source and target:
                rows = [row for row in self.corpus.references if row["paper_id"] == source["subject"] and row.get("resolved_paper_id") == target["subject"]]
                if not rows:
                    raise CorpusAbsent(f"No resolved in-corpus citation from {source['object']} to {target['object']}.", "citation_index", len(self.corpus.references))
                triples = [self._citation_triple(row) for row in rows]
                return self._fact_answer(triples, f"Yes. {source['object']} cites {target['object']} [{', '.join(item.source_id for item in triples)}].")

        years = [row for row in facts if row["predicate"] == "year" and str(row["object"]).isdigit()]
        if "how many papers" in question and "published" in question:
            match = re.search(r"\b(?:19|20)\d{2}\b", question)
            if match:
                rows = [row for row in years if str(row["object"]) == match.group()]
                if rows:
                    triples = [self._fact_triple(row) for row in rows]
                    return self._fact_answer(triples, f"{len({item.subject for item in triples})} corpus papers were published in {match.group()} [{', '.join(item.source_id for item in triples)}].")

        if years and ("earliest publication year" in question or "latest publication year" in question):
            year = min(int(row["object"]) for row in years) if "earliest" in question else max(int(row["object"]) for row in years)
            rows = [row for row in years if int(row["object"]) == year]
            triples = [self._fact_triple(row) for row in rows]
            names = [self.corpus.paper_title(item.subject) or item.subject for item in triples]
            return self._fact_answer(triples, f"The {'earliest' if 'earliest' in question else 'latest'} publication year is {year}: {', '.join(names)} [{', '.join(item.source_id for item in triples)}].")

        if "which author" in question and "most distinct papers" in question:
            grouped: dict[str, list[dict[str, Any]]] = {}
            for row in facts:
                if row["predicate"] == "author":
                    grouped.setdefault(str(row["object"]), []).append(row)
            distinct = {name: self._distinct_works(rows) for name, rows in grouped.items()}
            if distinct:
                maximum = max(len(rows) for rows in distinct.values())
                winners = {name: rows for name, rows in distinct.items() if len(rows) == maximum}
                triples = [self._fact_triple(row) for rows in winners.values() for row in rows]
                names = ", ".join(sorted(winners))
                return self._fact_answer(triples, f"{names} {'appears' if len(winners) == 1 else 'appear'} on the most distinct corpus papers ({maximum}) [{', '.join(item.source_id for item in triples)}].")

        return None

    def _distinct_works(self, author_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Treat indexed preprints and their near-identical journal versions as one work."""
        works: list[tuple[str, dict[str, Any]]] = []
        for row in sorted(author_rows, key=lambda item: str(item["subject"])):
            title = self.corpus.paper_title(str(row["subject"])) or str(row["subject"])
            normalized = " ".join(re.findall(r"[a-z0-9]+", title.casefold()))
            if any(SequenceMatcher(None, normalized, prior).ratio() >= 0.98 for prior, _ in works):
                continue
            works.append((normalized, row))
        return [row for _, row in works]

    def _find_title_fact(self, question: str) -> dict[str, Any] | None:
        terms = {word for word in re.findall(r"[a-z0-9]+", question) if len(word) > 2}
        candidates = [row for row in self.corpus.seed_facts() if row.get("predicate") == "title"]
        if not candidates:
            return None
        scored = [
            (len(terms & set(re.findall(r"[a-z0-9]+", str(row["object"]).casefold()))), row)
            for row in candidates
        ]
        score, row = max(scored, key=lambda item: item[0])
        return row if score >= 2 else None

    def _author_triples(self, paper_id: str) -> list[ExtractedTriple]:
        return [
            ExtractedTriple(
                subject=str(row["subject"]), predicate="authoredBy", object=str(row["object"]),
                source_id=str(row["source_id"]), evidence=f"Corpus fact {row.get('fact_id', row['source_id'])}.",
            )
            for row in self.corpus.seed_facts()
            if row.get("subject") == paper_id and row.get("predicate") == "author"
        ]

    def _direct_title_answer(self, state: BaselineState) -> AnswerResult | None:
        """Answer title lookup questions from verified metadata facts.

        The model can infer a broad ``Paper`` schema for a title question and then
        return an unhelpful comparison answer. Metadata facts are already verified
        during ingest, so using them here keeps the answer tied to a concrete row
        and preserves the normal provenance path.
        """

        question = state["question"].casefold()
        if "which paper" not in question and "what paper" not in question:
            return None
        terms = {word for word in re.findall(r"[a-z0-9]+", question) if len(word) > 2}
        candidates = [item for item in state["extracted"] if item.predicate == "title"]
        candidates.extend(
            ExtractedTriple(
                subject=str(row["subject"]),
                predicate="title",
                object=str(row["object"]),
                source_id=str(row["source_id"]),
                evidence=f"Corpus fact {row.get('fact_id', row['source_id'])}.",
            )
            for row in self.corpus.seed_facts()
            if row.get("predicate") == "title"
        )
        candidates = [
            item
            for item in candidates
            if len(terms & set(re.findall(r"[a-z0-9]+", item.object.casefold()))) >= 2
        ]
        if not candidates:
            return None
        selected = max(
            candidates,
            key=lambda item: len(terms & set(re.findall(r"[a-z0-9]+", item.object.casefold()))),
        )
        sources = sorted(
            {selected.source_id}
            | {item.source_id for item in state["extracted"] if item.subject == selected.subject}
        )
        canonical = self.corpus.canonical_paper_iri(selected.subject)
        return AnswerResult(
            sparql=(
                "PREFIX hrn: <https://harness.local/schema#>\n"
                "SELECT ?paper ?title WHERE { ?paper hrn:title ?title . }"
            ),
            answer=f"{selected.object} ({canonical or 'corpus paper'}) [{'; '.join(sources)}].",
            cited_source_ids=sources,
        )

    @staticmethod
    def needs_clarification(question: str) -> bool:
        text = question.casefold()
        return bool(re.search(r"\bbest (?:author|paper)\b|\bhardest subset\b", text))

    def _clarification(self, run_id: str, started_at: datetime, question: str, budget: str) -> BaselineResult:
        if "author" in question.casefold():
            options = (
                ("Most distinct papers", "Count distinct works authored in the indexed corpus."),
                ("Most in-corpus citations", "Count citations from other indexed papers."),
                ("Most reported benchmark results", "Compare supported benchmark claims in indexed papers."),
            )
        else:
            options = (
                ("Reported benchmark performance", "Compare results on a named benchmark and metric."),
                ("Methodological difficulty", "Compare the stated task and evaluation constraints."),
                ("In-corpus evidence coverage", "Compare how much direct evidence the corpus contains."),
            )
        options += (("Combine all applicable interpretations", "Report each applicable criterion separately with its evidence."),)
        readings = [
            Reading(
                label=label, gloss=gloss, classes=[], axioms=[], sparql="", answer="",
                triples=[], metrics=Metrics(), status="proposed",
            )
            for label, gloss in options
        ]
        self._emit("Clarification required", f"{len(readings)} criteria offered before retrieval")
        path = "baseline-enhanced" if self.enhanced else "baseline"
        run = Run(
            id=run_id, startedAt=started_at,
            durationS=round(time.monotonic() - self.started, 3),
            question=question, path=path, budget=budget,
            status="awaiting_reading", readings=readings, chosenReading=None,
            trace=self.trace, modelSmall=getattr(self.extraction.model, "model", "recorded"),
            artifacts=list(required_artifacts(path)),
        )
        artifacts = {name: b"" for name in required_artifacts(path) if name != "run.json"}
        artifacts["trace.json"] = json.dumps(
            [row.model_dump(by_alias=True) for row in self.trace], indent=2
        ).encode()
        artifacts["answer.md"] = b"Choose a criterion to continue the research."
        if self.enhanced:
            artifacts["retrieval.json"] = b'{"queries": [], "stop_reason": "awaiting_reading"}'
        return {"run": run, "artifacts": artifacts}

    def run(self, question: str, budget: str = "fast", run_id: str | None = None) -> BaselineResult:
        run_id = run_id or f"r-{uuid.uuid4().hex[:8]}"
        started_at = datetime.now(UTC)
        self.started = time.monotonic()
        self.current_run_id = run_id
        self.trace = []
        if self.needs_clarification(question):
            return self._clarification(run_id, started_at, question, budget)
        try:
            lookup_started = time.monotonic()
            direct, triples = self._direct_metadata_answer({"question": question, "extracted": []})
            if direct is not None and triples:
                properties = list(dict.fromkeys(item.predicate for item in triples))
                inferred = SchemaInference(
                    label="Corpus facts", gloss="Verified facts relevant to the question.",
                    classes=["Paper", "Author"] if "authoredBy" in properties else ["Paper"],
                    properties=properties,
                )
                self._emit("Retrieving verified corpus facts", f"{len(triples)} facts")
                self._emit("Selecting atomic triples", f"{len(triples)} question-relevant triples")
                state: BaselineState = {
                    "question": question, "passages": [], "inferred": inferred,
                    "extracted": triples, "answer": direct,
                    "retrieval": {
                        "queries": [], "stop_reason": "verified_facts",
                        "fact_lookup_ms": round((time.monotonic() - lookup_started) * 1000, 1),
                    },
                    "extraction_batches": [{"stage": "verified_facts", "triples": len(triples)}],
                }
            elif direct is not None:
                raise NoResults("No in-corpus evidence found for a supported answer.")
            else:
                state = self.graph.invoke({"question": question})
            run, artifacts = self._success(run_id, started_at, question, budget, state)
        except CorpusAbsent as exc:
            run, artifacts = self._absence(run_id, started_at, question, budget, exc)
        except NoEvidenceFound as exc:
            run, artifacts = self._no_evidence(run_id, started_at, question, budget, exc)
        except NoResults as exc:
            run, artifacts = self._failure(
                run_id, started_at, question, budget, "no_results", str(exc)
            )
        except ModelFailure as exc:
            run, artifacts = self._failure(
                run_id, started_at, question, budget, "provider_error", str(exc)
            )
        except ValueError as exc:
            reason = (
                "query_rejected"
                if "SPARQL" in str(exc) or "query" in str(exc)
                else "schema_rejected"
            )
            run, artifacts = self._failure(run_id, started_at, question, budget, reason, str(exc))
        return {"run": run, "artifacts": artifacts}

    def _failure(
        self,
        run_id: str,
        started_at: datetime,
        question: str,
        budget: str,
        reason: str,
        detail: str,
    ) -> tuple[Run, dict[str, bytes]]:
        self._emit("Run incomplete", f"{reason}: {detail}")
        run = Run(
            id=run_id,
            startedAt=started_at,
            durationS=round(time.monotonic() - self.started, 3),
            question=question,
            path="baseline-enhanced" if self.enhanced else "baseline",
            budget=budget,
            status="incomplete",
            reason=reason,
            trace=self.trace,
            modelAgent="",
            modelSmall=getattr(self.extraction.model, "model", "recorded"),
            artifacts=list(required_artifacts("baseline-enhanced" if self.enhanced else "baseline")),
        )
        path = "baseline-enhanced" if self.enhanced else "baseline"
        artifacts = {name: b"" for name in required_artifacts(path) if name != "run.json"}
        artifacts["trace.json"] = json.dumps(
            [r.model_dump(by_alias=True) for r in self.trace], indent=2
        ).encode()
        artifacts["answer.md"] = detail.encode()
        return run, artifacts

    def _absence(
        self, run_id: str, started_at: datetime, question: str, budget: str,
        absence: CorpusAbsent,
    ) -> tuple[Run, dict[str, bytes]]:
        self._emit("Corpus absence verified", str(absence))
        path = "baseline-enhanced" if self.enhanced else "baseline"
        coverage = {"exhaustive": True, "inspected": absence.inspected, "matches": 0, "failures": 0}
        run = Run(
            id=run_id, startedAt=started_at,
            durationS=round(time.monotonic() - self.started, 3),
            question=question, path=path, budget=budget, status="corpus_absent",
            reason=str(absence), evidenceSurface=absence.evidence_surface,
            searchCoverage=coverage, trace=self.trace,
            producingPath=path, executionStatus="completed",
            artifacts=list(required_artifacts(path)),
        )
        artifacts = {name: b"" for name in required_artifacts(path) if name != "run.json"}
        artifacts["answer.md"] = str(absence).encode()
        artifacts["claims.json"] = b'{"claims": [], "relations": []}'
        artifacts["coverage_plan.json"] = json.dumps({
            "evidence_surface": absence.evidence_surface, **coverage,
        }).encode()
        artifacts["trace.json"] = json.dumps(
            [row.model_dump(by_alias=True) for row in self.trace], indent=2,
        ).encode()
        return run, artifacts

    def _no_evidence(
        self, run_id: str, started_at: datetime, question: str, budget: str,
        absence: NoEvidenceFound,
    ) -> tuple[Run, dict[str, bytes]]:
        self._emit("Semantic search completed", str(absence))
        path = "baseline-enhanced" if self.enhanced else "baseline"
        run = Run(
            id=run_id, startedAt=started_at, durationS=round(time.monotonic() - self.started, 3),
            question=question, path=path, budget=budget, status="no_evidence_found",
            reason=str(absence), evidenceSurface="indexed_full_text", searchCoverage=absence.coverage,
            trace=self.trace, producingPath=path, executionStatus="completed",
            artifacts=list(required_artifacts(path)),
        )
        artifacts = {name: b"" for name in required_artifacts(path) if name != "run.json"}
        artifacts["answer.md"] = str(absence).encode()
        artifacts["claims.json"] = b'{"claims": [], "relations": []}'
        artifacts["coverage_plan.json"] = json.dumps({"evidence_surface": "indexed_full_text", **absence.coverage}).encode()
        artifacts["trace.json"] = json.dumps([row.model_dump(by_alias=True) for row in self.trace], indent=2).encode()
        return run, artifacts

    def _success(
        self,
        run_id: str,
        started_at: datetime,
        question: str,
        budget: str,
        state: BaselineState,
    ) -> tuple[Run, dict[str, bytes]]:
        inferred = state["inferred"]
        extracted = state["extracted"]
        answer = state["answer"]
        schema = self.extraction.schema
        path = "baseline-enhanced" if self.enhanced else "baseline"
        source_ids = {row["passage_id"] for row in state["passages"]}
        source_ids |= {
            str(row.get("source_id")) for row in self.corpus.fact_rows if row.get("source_id")
        }
        source_ids |= {str(row["reference_id"]) for row in self.corpus.references}
        dataset = ProvenanceDataset(schema, source_ids)
        ui_triples: list[Triple] = []
        seed_rows = self.corpus.seed_facts()
        seed_rdf: list[tuple[Any, URIRef, Any]] = []
        for row in seed_rows:
            raw_predicate = str(row["predicate"])
            predicate_name = {"author": "authoredBy"}.get(raw_predicate, raw_predicate)
            if raw_predicate == "cites" and not self.corpus.canonical_paper_iri(str(row.get("object", ""))):
                title = self.corpus.reference_title(str(row.get("source_id", "")))
                if not title:
                    continue
                predicate_name = "citesTitle"
            if not hasattr(HRN, predicate_name):
                continue
            local_predicate = URIRef(HRN[predicate_name])
            predicate = URIRef(
                {
                    "title": DBLP.title,
                    "authoredBy": DBLP.authoredBy,
                    "year": DBLP.yearOfPublication,
                    "venue": DBLP.publishedIn,
                }.get(predicate_name, local_predicate)
            )
            subject_value = self.corpus.canonical_paper_iri(str(row["subject"])) or str(row["subject"])
            subject = URIRef(
                subject_value
                if subject_value.startswith("https://dblp.org/rec/")
                else f"https://harness.local/resource/{quote(subject_value, safe='')}"
            )
            expected = schema.graph.value(predicate, RDFS.range) or schema.graph.value(local_predicate, RDFS.range)
            raw = title if predicate_name == "citesTitle" else row.get("object")
            target_iri = self.corpus.canonical_paper_iri(str(raw))
            object_ = Literal(raw) if predicate_name == "citesTitle" or (expected and str(expected).startswith(str(XSD))) else URIRef(
                target_iri
                or f"https://harness.local/resource/{quote(str(raw), safe='')}"
            )
            seed_rdf.append((subject, predicate, object_))
        if seed_rdf:
            dataset.add_event("corpus-seed", seed_rdf, sorted({str(row["source_id"]) for row in seed_rows}))
        for index, item in enumerate(extracted, start=1):
            candidate = Triple(
                s=item.subject, p=item.predicate, o=item.object,
                support="1 passage", evidence=item.evidence, source=item.source_id,
            )
            if not valid_edge(candidate):
                continue
            local_predicate = URIRef(HRN[item.predicate])
            predicate = URIRef(
                {
                    "title": DBLP.title,
                    "authoredBy": DBLP.authoredBy,
                    "year": DBLP.yearOfPublication,
                    "venue": DBLP.publishedIn,
                }.get(item.predicate, local_predicate)
            )
            expected = schema.graph.value(predicate, RDFS.range) or schema.graph.value(local_predicate, RDFS.range)
            subject_iri = self.corpus.canonical_paper_iri(item.subject)
            object_iri = self.corpus.canonical_paper_iri(item.object)
            if (
                expected and str(expected) == str(HRN.Paper)
                and not object_iri and item.object not in self.corpus.by_paper
            ):
                # A paper-valued relation must resolve to an indexed paper.
                continue
            object_ = (
                Literal(item.object)
                if expected and str(expected).startswith(str(XSD))
                else URIRef(object_iri or f"https://harness.local/resource/{quote(item.object, safe='')}")
            )
            subject = URIRef(
                subject_iri
                or f"https://harness.local/resource/{quote(item.subject, safe='')}"
            )
            dataset.add_event(f"extract-{index}", [(subject, predicate, object_)], [item.source_id])
            ui_triples.append(
                Triple(
                    s=subject_iri or item.subject,
                    p=item.predicate,
                    o=object_iri or item.object,
                    support="1 passage",
                    evidence=item.evidence,
                    source=item.source_id,
                    tripleId=f"t-{index:03d}",
                )
            )
        def paper_for_source(source_id: str, subject: str) -> str:
            row = self.corpus.by_passage.get(source_id)
            paper_id = str(row["paper_id"]) if row else subject
            return self.corpus.canonical_paper_iri(paper_id) or paper_id

        claims, claim_relations = build_claims(ui_triples, paper_for_source)
        if state["passages"] and any(claim.kind == "ResultClaim" for claim in claims):
            try:
                claims = enrich_result_claims(
                    claims, state["passages"], self.refinement_model or self.extraction.model,
                )
                self._emit("Enriching claim context", f"{len(claims)} provenance-bearing claims")
            except (ModelFailure, ValueError, TypeError) as exc:
                self._emit("Claim context remains partial", type(exc).__name__)
        claim_relations = classify_claim_relations(claims)
        query_claim = derive_query_claim(question, claims)
        if query_claim is not None:
            claims.append(query_claim)
        if state.get("retrieval", {}).get("stop_reason") != "verified_facts" and claims:
            grouped: dict[str, list[str]] = {}
            for claim in claims:
                if "object" not in claim.fields:
                    continue
                title = self.corpus.paper_title(claim.paper_id) or claim.paper_id
                statement = claim.fields["object"].value
                grouped.setdefault(title, []).append(
                    f"{statement} [{', '.join(claim.source_ids)}]"
                )
            answer = answer.model_copy(update={
                "answer": "The indexed papers report: " + " ".join(
                    f"{title}: {'; '.join(statements)}."
                    for title, statements in grouped.items()
                ),
                "cited_source_ids": sorted({source for claim in claims for source in claim.source_ids}),
            })
        reading = Reading(
            label=inferred.label,
            gloss=inferred.gloss,
            classes=inferred.classes,
            properties=inferred.properties,
            axioms=self.extraction.axioms(inferred),
            sparql=answer.sparql,
            answer=answer.answer,
            triples=ui_triples,
            claims=claims,
            claimRelations=claim_relations,
            # The prose renderer below only enumerates these source-backed claims.
            answerClaimIds=[claim.id for claim in claims],
            metrics=Metrics(),
            status="answered",
        )
        run = Run(
            id=run_id,
            startedAt=started_at,
            durationS=round(time.monotonic() - self.started, 3),
            question=question,
            path=path,
            budget=budget,
            status="answered",
            producingPath=path,
            executionStatus="completed",
            readings=[reading],
            chosenReading=0,
            trace=self.trace,
            coverage=None,
            modelAgent="",
            modelSmall=getattr(self.extraction.model, "model", "recorded"),
            artifacts=list(required_artifacts(path)),
        )
        report = validate_subschema(schema.graph)
        artifacts = {
            "subschema.ttl": schema.graph.serialize(format="turtle").encode(),
            "query.rq": answer.sparql.encode(),
            "coverage_plan.json": json.dumps(
                {"coverage": None, "reason": "baseline has no ledger"}, indent=2
            ).encode(),
            "ledger.json": json.dumps({"available": False}, indent=2).encode(),
            "graph.trig": dataset.export_trig().encode(),
            "triples.ttl": dataset.export_turtle().encode(),
            "claims.json": json.dumps({
                "claims": [claim.model_dump(by_alias=True) for claim in claims],
                "relations": [relation.model_dump(by_alias=True) for relation in claim_relations],
            }, ensure_ascii=False, indent=2).encode(),
            "shacl_report.ttl": report.graph.serialize(format="turtle").encode(),
            "answer.md": answer.answer.encode(),
            "citations.json": json.dumps(answer.cited_source_ids, indent=2).encode(),
            "trace.json": json.dumps(
                [r.model_dump(by_alias=True) for r in self.trace], indent=2
            ).encode(),
        }
        tool_calls = self._workflow_tool_calls(state, run_id)
        artifacts["tool_calls.jsonl"] = b"\n".join(
            json.dumps(item, ensure_ascii=False, default=str).encode() for item in tool_calls
        ) + (b"\n" if tool_calls else b"")
        if self.enhanced:
            artifacts["retrieval.json"] = json.dumps(state.get("retrieval", {}), indent=2).encode()
            artifacts["extraction_batches.json"] = json.dumps(state.get("extraction_batches", []), indent=2).encode()
            artifacts["seed_manifest.json"] = json.dumps(
                {
                    "facts": len(seed_rows),
                    "seed_graph": "corpus-seed",
                    "ontology": "dblp_schema_2024-06-14.ttl",
                    "ontology_version": schema.seed_schema_version,
                    "ontology_sha256": schema.seed_schema_sha256,
                },
                indent=2,
            ).encode()
        return run, artifacts

    def _workflow_tool_calls(self, state: BaselineState, run_id: str) -> list[dict[str, Any]]:
        """Record only corpus operations that the local workflow actually ran."""
        records: list[dict[str, Any]] = []
        retrieval = state.get("retrieval", {})
        queries = retrieval.get("queries", [])
        for index, query in enumerate(queries, start=1):
            record = {
                "call_id": f"tool-baseline-{index:03d}",
                "agent": "baseline",
                "tool": "search",
                "transport": "local",
                "arguments": {"dsl": query.get("query", ""), "limit": query.get("limit")},
                "result": {"passage_ids": query.get("passage_ids", [])},
                "result_count": query.get("returned", 0),
                "duration_ms": query.get("duration_ms", 0.0),
                "status": "ok",
            }
            records.append(record)
            record_tool_call(record, run_id=run_id)
        ranking = retrieval.get("ranking")
        if ranking:
            record = {
                "call_id": f"tool-baseline-{len(records) + 1:03d}",
                "agent": "baseline",
                "tool": "rank_evidence",
                "transport": "local",
                "arguments": {
                    "query": ranking["query"],
                    "passage_ids": ranking["candidate_ids"],
                },
                "result": {"passage_ids": ranking["passage_ids"]},
                "result_count": len(ranking["passage_ids"]),
                "duration_ms": ranking["duration_ms"],
                "status": "ok",
            }
            records.append(record)
            record_tool_call(record, run_id=run_id)
        if retrieval.get("stop_reason") == "verified_facts":
            record = {
                "call_id": f"tool-baseline-{len(records) + 1:03d}",
                "agent": "baseline",
                "tool": "facts",
                "transport": "local",
                "arguments": {"question": state["question"]},
                "result": [item.model_dump() for item in state.get("extracted", [])],
                "result_count": len(state.get("extracted", [])),
                "duration_ms": retrieval.get("fact_lookup_ms", 0.0),
                "status": "ok",
            }
            records.append(record)
            record_tool_call(record, run_id=run_id)
        return records


__all__ = ["Baseline20", "BaselineResult", "BaselineState", "NoResults", "retrieval_dsl"]
