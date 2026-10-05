"""Run outcomes: turn a finished pipeline state into a Run and its artifacts.

A mixin of ``Baseline20``: methods use the pipeline's state (``self``).
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any

from harness_config import settings
from harness_extraction import (
    ModelFailure,
)
from harness_extraction.claims import (
    build_claims,
    classify_claim_relations,
    derive_query_claim,
    enrich_result_claims,
)
from harness_graphstore import ProvenanceDataset
from harness_observability import record_tool_call
from harness_schema.models import (
    AnswerSentence,
    Metrics,
    Reading,
    Run,
    Triple,
    required_artifacts,
)
from harness_semantic import validate_subschema
from rdflib import Graph

from . import answer_graph
from .backbone import answer_paper_keys, backbone_triples, origin_of, schema_terms, subschema_slice
from .run_graph import add_extracted, seed_rdf
from .state import (
    BaselineState,
    CorpusAbsent,
    NoEvidenceFound,
    NoResults,
)


class RunOutcomes:
    @property
    def path_name(self) -> str:
        """The run's path; the external harness overrides it with "external"."""
        return "baseline-enhanced" if self.enhanced else "baseline"

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
            path=self.path_name,
            budget=budget,
            status="incomplete",
            reason=reason,
            trace=self.trace,
            modelAgent="",
            modelSmall=getattr(self.extraction.model, "model", "recorded"),
            artifacts=list(required_artifacts(self.path_name)),
        )
        path = self.path_name
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
        path = self.path_name
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
        path = self.path_name
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

    def _schema_check(self, conforms: bool, terms: list, turtle: str) -> None:
        """Phase 5.7: the run's own schema and export, as a stage check (graph-shape counts the same)."""
        invented = [term for term in terms if term.origin == "invented"]
        properties = [term for term in invented if term.kind != "class"]
        problems = []
        if not conforms:
            problems.append("SHACL shapes do not conform (see shacl_report.ttl)")
        unparented = [term.name for term in invented if not term.parent]
        if unparented:
            problems.append(f"{len(unparented)} invented terms without a parent: {', '.join(unparented[:3])}")
        untyped = [term.name for term in properties if not (term.domain and term.range)]
        if untyped:
            problems.append(f"{len(untyped)} invented properties without domain and range: {', '.join(untyped[:3])}")
        try:
            exported = len(Graph().parse(data=turtle, format="turtle"))
        except Exception as exc:  # noqa: BLE001 - any parser error means the export is invalid
            problems.append(f"triples.ttl does not parse ({type(exc).__name__})")
            exported = 0
        self._check("schema", "warn" if problems else "pass", "; ".join(problems) or (
            f"{len(terms)} OWL 2 terms ({len(properties)} invented properties and "
            f"{len(invented) - len(properties)} invented classes, each with a parent); SHACL conforms; "
            f"triples.ttl parses ({exported} triples)"))

    def _success(
        self,
        run_id: str,
        started_at: datetime,
        question: str,
        budget: str,
        state: BaselineState,
    ) -> tuple[Run, dict[str, bytes]]:
        """The answered Run and its artifacts: graph, claims, answer text, checks, files."""
        inferred, extracted, schema = state["inferred"], state["extracted"], self.extraction.schema
        backbone_on = settings().dblp_backbone
        # The papers this answer is about: the export and the backbone are limited to them.
        answer_keys = answer_paper_keys(self.corpus, [item.subject for item in extracted],
                                        [item.object for item in extracted],
                                        [item.source_id for item in extracted]) if backbone_on else []
        dataset, seed_rows = self._provenance(state, answer_keys if backbone_on else None)
        ui_triples = self._answer_triples(dataset, extracted)
        claims, claim_relations = self._claims(question, ui_triples, state["passages"])
        answer = self._answer_text(state, state["answer"], ui_triples, claims)
        from_facts = state.get("retrieval", {}).get("stop_reason") == "verified_facts"
        backbone: list[Triple] = []
        if backbone_on and not from_facts:
            papers = answer_paper_keys(self.corpus, [*answer_keys, *(claim.paper_id for claim in claims)])
            backbone = backbone_triples(self.corpus, papers)
            if backbone:
                self._emit("Adding DBLP backbone", f"{len(backbone)} author, venue and citation facts "
                           f"for {len(papers)} papers")
        graph = answer_graph.build(
            ui_triples + backbone, claims, state.get("computed") or [],
            is_paper=lambda value: bool(self.corpus.canonical_paper_iri(value)) or value in self.corpus.by_paper,
            paper_label=lambda value: self.corpus.paper_title(
                self.corpus.by_dblp.get(value, value)) or value,
            origin_of=lambda name: origin_of(schema.graph, name),
        )
        terms = schema_terms(schema.graph, inferred.classes, inferred.properties,
                             sorted({triple.p for triple in [*ui_triples, *backbone]}))
        self._answer_checks(ui_triples, answer, from_facts)
        plan = state.get("plan")
        reading = Reading(
            measure=plan.stated_measure() if plan is not None else None,
            graph=graph,
            label=inferred.label,
            gloss=inferred.gloss,
            classes=inferred.classes,
            properties=inferred.properties,
            axioms=self.extraction.axioms(inferred),
            sparql=answer.sparql,
            answer=answer.answer,
            triples=ui_triples,
            claims=claims,
            schemaTerms=terms,
            answerSentences=[AnswerSentence.model_validate(item) for item in answer.sentences],
            claimRelations=claim_relations,
            # The prose renderer below only enumerates these source-backed claims.
            answerClaimIds=[claim.id for claim in claims],
            metrics=Metrics(),
            status="answered",
        )
        path = self.path_name
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
        self._schema_check(report.conforms, terms, dataset.export_turtle())
        artifacts = self._artifacts(run_id, state, answer, claims, claim_relations, terms, dataset, report)
        if self.enhanced:
            artifacts["seed_manifest.json"] = json.dumps(
                {
                    "facts": len(seed_rows),
                    "seed_graph": "corpus-seed",
                    "ontology": "dblp_schema_2024-06-14.ttl",
                    "ontology_version": schema.seed_schema_version,
                    "ontology_sha256": schema.seed_schema_sha256,
                    "content_core": schema.content_core,
                    "content_core_sha256": schema.content_core_sha256,
                },
                indent=2,
            ).encode()
        return run, artifacts

    # ─── Graph ───────────────────────────────────────────────────────────────

    def _provenance(self, state: BaselineState, answer_keys: list[str] | None
                    ) -> tuple[ProvenanceDataset, list[dict[str, Any]]]:
        """The run's provenance dataset with the seed facts (only the answer's papers when
        ``answer_keys`` is given) as one event; returns it and the seed rows used."""
        source_ids = {row["passage_id"] for row in state["passages"]}
        source_ids |= {str(row.get("source_id")) for row in self.corpus.fact_rows if row.get("source_id")}
        source_ids |= {str(row["reference_id"]) for row in self.corpus.references}
        dataset = ProvenanceDataset(self.extraction.schema, source_ids)
        seed_rows = self.corpus.seed_facts()
        if answer_keys is not None:
            wanted = set(answer_keys)
            seed_rows = [row for row in seed_rows if str(row.get("subject")) in wanted
                         and (row.get("predicate") != "cites" or str(row.get("object")) in wanted)]
        rdf = seed_rdf(self.corpus, self.extraction.schema, seed_rows)
        if rdf:
            dataset.add_event("corpus-seed", rdf, sorted({str(row["source_id"]) for row in seed_rows}))
        return dataset, seed_rows

    def _answer_triples(self, dataset: ProvenanceDataset, extracted: list[Any]) -> list[Triple]:
        ui_triples, dropped = add_extracted(dataset, self.corpus, self.extraction.schema, extracted)
        lost = ", ".join(f"{count} {why}" for why, count in dropped.items() if count)
        if not ui_triples:
            # Nothing the answer could cite reached the graph: an honest "no results", not an answer
            # without facts.
            raise NoResults(f"none of the {len(extracted)} extracted triples could enter the answer graph ({lost}); "
                            "a paper-valued relation must name an indexed paper")
        if lost:
            self._emit("Graph building dropped triples", f"{lost} of {len(extracted)}")
        return ui_triples

    # ─── Claims and answer text ──────────────────────────────────────────────

    def _claims(self, question: str, ui_triples: list[Triple], passages: list[dict[str, Any]]
                ) -> tuple[list[Any], list[Any]]:
        def paper_for_source(source_id: str, subject: str) -> str:
            row = self.corpus.by_passage.get(source_id)
            paper_id = str(row["paper_id"]) if row else subject
            return self.corpus.canonical_paper_iri(paper_id) or paper_id

        claims, _ = build_claims(ui_triples, paper_for_source)
        if passages and any(claim.kind == "ResultClaim" for claim in claims):
            try:
                claims = enrich_result_claims(claims, passages, self.refinement_model or self.extraction.model)
                self._emit("Enriching claim context", f"{len(claims)} provenance-bearing claims")
            except (ModelFailure, ValueError, TypeError) as exc:
                self._emit("Claim context remains partial", type(exc).__name__)
        claim_relations = classify_claim_relations(claims)
        query_claim = derive_query_claim(question, claims)
        if query_claim is not None:
            claims.append(query_claim)
        return claims, claim_relations

    def _answer_text(self, state: BaselineState, answer: Any, ui_triples: list[Triple], claims: list[Any]) -> Any:
        """Drop sentences whose triples missed the graph; an unverified (Fast) answer becomes the
        list of source-backed claims. A verified answer (sentences with verdicts) is shown as written."""
        if answer.sentences:
            kept_sources = {triple.source for triple in ui_triples}
            sentences = [item if item["verdict"] == "removed" or set(item["sourceIds"]) & kept_sources
                         else {**item, "verdict": "removed", "reason": "its triples did not pass graph building"}
                         for item in answer.sentences]
            stated = [item for item in sentences if item["verdict"] != "removed"]
            if stated:
                answer = answer.model_copy(update={
                    "sentences": sentences, "answer": " ".join(item["text"] for item in stated),
                    "cited_source_ids": list(dict.fromkeys(s for item in stated for s in item["sourceIds"]))})
            else:
                answer = answer.model_copy(update={"sentences": []})     # falls back to the claim list below
        if state.get("retrieval", {}).get("stop_reason") in {"verified_facts", "external"} or not claims \
                or answer.sentences:
            return answer
        grouped: dict[str, list[str]] = {}
        for claim in claims:
            if "object" not in claim.fields:
                continue
            title = self.corpus.paper_title(claim.paper_id) or claim.paper_id
            grouped.setdefault(title, []).append(f"{claim.fields['object'].value} [{', '.join(claim.source_ids)}]")
        return answer.model_copy(update={
            "answer": "The indexed papers report: " + " ".join(
                f"{title}: {'; '.join(statements)}." for title, statements in grouped.items()),
            "cited_source_ids": sorted({source for claim in claims for source in claim.source_ids}),
        })

    # ─── Checks and files ────────────────────────────────────────────────────

    def _answer_checks(self, ui_triples: list[Triple], answer: Any, from_facts: bool) -> None:
        supported = bool(ui_triples) and all(t.source.strip() and t.evidence.strip() for t in ui_triples)
        self._check("triple provenance", "pass" if supported else "fail",
                    f"{len(ui_triples)} triples, each with a source and evidence" if supported
                    else "some triples lack a source or evidence")
        if not self.refine:
            return
        if from_facts:
            self._check("answer verification", "skipped", "answer composed directly from verified facts")
        elif any(item.get("verdict") == "from_facts" for item in answer.sentences):
            self._check("answer verification", "warn", "no composed sentence passed; the answer states the "
                        "extracted facts, which evidence review kept")
        else:
            self._check("answer verification", "pass", "every sentence in the answer was checked against its triples")

    def _artifacts(self, run_id: str, state: BaselineState, answer: Any, claims: list[Any],
                   claim_relations: list[Any], terms: list[Any], dataset: ProvenanceDataset,
                   report: Any) -> dict[str, bytes]:
        schema = self.extraction.schema
        artifacts = {
            # The question's own terms with their parents, not the whole seed ontology.
            "subschema.ttl": subschema_slice(schema.graph, terms).serialize(format="turtle").encode(),
            "query.rq": answer.sparql.encode(),
            "coverage_plan.json": json.dumps(
                {"coverage": None, "reason": "baseline has no ledger"}, indent=2
            ).encode(),
            # Shared downstream contract: controllers supply evidence; claim
            # construction, validation, provenance, and prose consume this
            # same package for Fast, Adaptive, and Agentic paths.
            "ledger.json": json.dumps({
                "evidence_bundle": {
                    "passage_ids": [str(item["passage_id"]) for item in state.get("passages", [])],
                    "source_ids": sorted({source for claim in claims for source in claim.source_ids}),
                    "coverage": state.get("retrieval", {}),
                },
                "claim_subgraph_ids": [claim.id for claim in claims],
                "execution_metadata": {
                    "controller": {"baseline-enhanced": "adaptive", "baseline": "fast"}.get(self.path_name, self.path_name),
                    "run_id": run_id,
                },
            }, ensure_ascii=False, indent=2).encode(),
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
            artifacts["evidence_graph.json"] = json.dumps(evidence_graph(state, state["inferred"]),
                                                          ensure_ascii=False, indent=2).encode()
        return artifacts

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


def evidence_graph(state: BaselineState, inferred: Any) -> dict[str, Any]:
    """Everything the run extracted, not only what the answer used.

    ``triples``: every extracted triple with the evidence review's verdict (``relevant``: true, false,
    or null when there was no review) and its reason. ``unmapped_dropped``: open facts lost because their
    relation fit no subschema term (only with HARNESS_UNMAPPED_RELATIONS=drop). Paths that answer from
    verified facts have no extraction; their evidence graph is empty.
    """
    rows = list(state.get("evidence_graph") or [])
    unmapped = [triple for batch in state.get("extraction_batches") or [] if isinstance(batch, dict)
                for report in batch.get("edc") or [] if isinstance(report, dict)
                and report.get("stage") == "unmapped_dropped" for triple in report.get("triples") or []]
    config = settings()
    return {
        "properties": list(getattr(inferred, "properties", []) or []),
        "classes": list(getattr(inferred, "classes", []) or []),
        "counts": {
            "triples": len(rows),
            "relevant": sum(1 for row in rows if row.get("relevant") is True),
            "not_relevant": sum(1 for row in rows if row.get("relevant") is False),
            "unreviewed": sum(1 for row in rows if row.get("relevant") is None),
            "unmapped_dropped": len(unmapped),
        },
        "settings": {"schema_scope": config.schema_scope, "graph_linking": config.graph_linking,
                     "unmapped_relations": config.unmapped_relations, "profile": config.profile},
        "triples": rows,
        "unmapped_dropped": unmapped,
    }

