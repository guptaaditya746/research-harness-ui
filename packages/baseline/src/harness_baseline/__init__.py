"""Baseline-20 as a fixed four-node LangGraph workflow."""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
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
from harness_graphstore import ProvenanceDataset, sparql_check
from harness_schema.models import Metrics, Reading, Run, TraceRow, Triple, required_artifacts
from harness_semantic import HRN, validate_subschema
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
        trace_sink: TraceSink | None = None,
    ) -> None:
        self.corpus = corpus
        self.extraction = extraction
        self.retrieve_k = retrieve_k
        self.enhanced = enhanced
        self.trace_sink = trace_sink or (lambda _: None)
        self.started = 0.0
        self.trace: list[TraceRow] = []
        self.graph = self._graph()

    def _emit(self, action: str, returned: str) -> None:
        row = TraceRow(
            atS=round(time.monotonic() - self.started, 3), action=action, returned=returned
        )
        self.trace.append(row)
        self.trace_sink(row)

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
        dsl = retrieval_dsl(state["question"])
        queries = [dsl]
        if self.enhanced:
            terms = [t for t in re.findall(r"[A-Za-z][A-Za-z0-9_-]+", state["question"]) if len(t) >= 4]
            phrases = [f'"{" ".join(terms[i : i + 2])}"' for i in range(max(0, len(terms) - 1))]
            queries.extend(phrases[:4])
        found: dict[str, dict[str, Any]] = {}
        rounds: list[dict[str, Any]] = []
        for round_no, query in enumerate(queries, start=1):
            hits = self.corpus.search(query, limit=self.retrieve_k)["items"]
            before = len(found)
            found.update({row["passage_id"]: row for row in hits})
            rounds.append({"round": round_no, "query": query, "returned": len(hits), "new": len(found) - before})
            if len(found) >= self.retrieve_k:
                break
        rows = list(found.values())[: self.retrieve_k]
        if not rows:
            raise NoResults("retrieval returned no passages")
        self._emit("Retrieving top passages", f"{len(rows)} passages across {len(rounds)} rounds")
        return {"passages": rows, "retrieval": {"queries": rounds, "stop_reason": "coverage_or_limit"}}

    def _infer_schema(self, state: BaselineState) -> dict[str, Any]:
        inferred = self.extraction.infer_schema(state["question"], state["passages"])
        self._emit(
            "Inferring schema",
            f"{len(inferred.classes)} classes · {len(inferred.properties)} properties",
        )
        return {"inferred": inferred}

    def _extract_build(self, state: BaselineState) -> dict[str, Any]:
        if self.enhanced:
            extracted, reports = self.extraction.extract_batches(
                state["question"], state["inferred"], state["passages"]
            )
        else:
            extracted = self.extraction.extract(
                state["question"], state["inferred"], state["passages"]
            ).triples
            reports = []
        if not extracted:
            raise NoResults("extractor returned no supported triples")
        self._emit("Extracting triples", f"{len(extracted)} supported triples")
        return {"extracted": extracted, "extraction_batches": reports}

    def _query_answer(self, state: BaselineState) -> dict[str, Any]:
        answer = self.extraction.answer(state["question"], state["inferred"], state["extracted"])
        answer = answer.model_copy(
            update={"sparql": sparql_check(answer.sparql, self.extraction.schema)}
        )
        self._emit(
            "Querying graph and composing answer", f"{len(answer.cited_source_ids)} cited passages"
        )
        return {"answer": answer}

    def run(self, question: str, budget: str = "fast", run_id: str | None = None) -> BaselineResult:
        run_id = run_id or f"r-{uuid.uuid4().hex[:8]}"
        started_at = datetime.now(UTC)
        self.started = time.monotonic()
        self.trace = []
        try:
            state = self.graph.invoke({"question": question})
            run, artifacts = self._success(run_id, started_at, question, budget, state)
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
        dataset = ProvenanceDataset(schema, source_ids)
        ui_triples: list[Triple] = []
        seed_rows = self.corpus.seed_facts()
        seed_rdf: list[tuple[Any, URIRef, Any]] = []
        for row in seed_rows:
            predicate_name = {"author": "authoredBy"}.get(row["predicate"], row["predicate"])
            if not hasattr(HRN, predicate_name):
                continue
            predicate = URIRef(HRN[predicate_name])
            subject = URIRef(f"https://harness.local/resource/{quote(str(row['subject']), safe='')}")
            expected = schema.graph.value(predicate, RDFS.range)
            raw = row.get("object")
            object_ = Literal(raw) if expected and str(expected).startswith(str(XSD)) else URIRef(
                f"https://harness.local/resource/{quote(str(raw), safe='')}"
            )
            seed_rdf.append((subject, predicate, object_))
        if seed_rdf:
            dataset.add_event("corpus-seed", seed_rdf, sorted({str(row["source_id"]) for row in seed_rows}))
        for index, item in enumerate(extracted, start=1):
            predicate = URIRef(HRN[item.predicate])
            expected = schema.graph.value(predicate, RDFS.range)
            object_ = (
                Literal(item.object)
                if expected and str(expected).startswith(str(XSD))
                else URIRef(f"https://harness.local/resource/{quote(item.object, safe='')}")
            )
            subject = URIRef(f"https://harness.local/resource/{quote(item.subject, safe='')}")
            dataset.add_event(f"extract-{index}", [(subject, predicate, object_)], [item.source_id])
            ui_triples.append(
                Triple(
                    s=item.subject,
                    p=item.predicate,
                    o=item.object,
                    support="1 passage",
                    evidence=item.evidence,
                    source=item.source_id,
                    tripleId=f"t-{index:03d}",
                )
            )
        reading = Reading(
            label=inferred.label,
            gloss=inferred.gloss,
            classes=inferred.classes,
            properties=inferred.properties,
            axioms=self.extraction.axioms(inferred),
            sparql=answer.sparql,
            answer=answer.answer,
            triples=ui_triples,
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
            "shacl_report.ttl": report.graph.serialize(format="turtle").encode(),
            "answer.md": answer.answer.encode(),
            "citations.json": json.dumps(answer.cited_source_ids, indent=2).encode(),
            "trace.json": json.dumps(
                [r.model_dump(by_alias=True) for r in self.trace], indent=2
            ).encode(),
        }
        if self.enhanced:
            artifacts["retrieval.json"] = json.dumps(state.get("retrieval", {}), indent=2).encode()
            artifacts["extraction_batches.json"] = json.dumps(state.get("extraction_batches", []), indent=2).encode()
            artifacts["seed_manifest.json"] = json.dumps(
                {"facts": len(seed_rows), "seed_graph": "corpus-seed"}, indent=2
            ).encode()
        return run, artifacts


__all__ = ["Baseline20", "BaselineResult", "BaselineState", "NoResults", "retrieval_dsl"]
