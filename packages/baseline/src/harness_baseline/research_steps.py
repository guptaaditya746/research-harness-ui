"""Research steps after retrieval: schema inference, extraction and claims, answer.

A mixin of ``Baseline20``: methods use the pipeline's state (``self``).
"""

from __future__ import annotations

import re
from typing import Any

from harness_config import settings
from harness_extraction import (
    AnswerResult,
    ExtractedTriple,
    UnsupportedAnswer,
    parallel_map,
)
from harness_graphstore import sparql_check
from rdflib import RDFS, XSD

from .state import (
    BaselineState,
    NoResults,
)


class ResearchSteps:
    def _infer_schema(self, state: BaselineState) -> dict[str, Any]:
        self.stage = "schema inference"
        seed_properties = set(self.extraction.property_names)
        notes = self._research_notes(state)
        inferred = self.extraction.infer_schema(self._research_question(state), state["passages"],
                                                **({"notes": notes} if notes else {}))
        for name in inferred.properties:
            if name in seed_properties or self.relation_proposer is None:
                continue
            predicate = self.extraction.property_names[name]
            graph = self.extraction.schema.graph
            domain = graph.value(predicate, RDFS.domain)
            range_ = graph.value(predicate, RDFS.range)
            definition = str(graph.value(predicate, RDFS.comment) or name)
            range_name = (
                f"xsd:{str(range_).rsplit('#', 1)[-1]}"
                if range_ is not None and str(range_).startswith(str(XSD))
                else str(range_).rsplit("#", 1)[-1] if range_ is not None else "xsd:string"
            )
            self.relation_proposer(
                name, definition, self.current_run_id,
                str(domain).rsplit("#", 1)[-1] if domain is not None else "Paper",
                range_name,
            )
        self._emit(
            "Inferring schema",
            f"{len(inferred.classes)} classes · {len(inferred.properties)} properties",
        )
        return {"inferred": inferred}

    def _extract_build(self, state: BaselineState) -> dict[str, Any]:
        self._check("schema inference", "pass", f"{', '.join(state['inferred'].classes)}")
        self.stage = "extraction"
        if self.edc:
            extracted = []
            reports = []
            # Small evidence batches keep the structured response focused and
            # avoid a single long generation blocking the whole run.
            # Batches run in parallel (HARNESS_EXTRACTION_CONCURRENCY); results keep batch order.
            batch_size = settings().edc_batch_passages
            batches = [state["passages"][start : start + batch_size]
                       for start in range(0, len(state["passages"]), batch_size)]
            workers = min(settings().extraction_concurrency, max(1, len(batches)))
            question = self._research_question(state)
            self._emit("Extracting passages", f"{len(state['passages'])} passages in {len(batches)} batches, "
                                              f"{workers} at a time")

            def run_batch(numbered: tuple[int, list[dict[str, Any]]]) -> tuple[Any, list[dict[str, Any]]]:
                batch_no, batch = numbered
                outcome = self.extraction.extract_edc(question, state["inferred"], batch)
                self._emit("Extracting passages", f"batch {batch_no}/{len(batches)} done · "
                                                  f"{len(outcome[0].triples)} triples")
                return outcome

            results = parallel_map(run_batch, list(enumerate(batches, start=1)), workers)
            for batch_no, (batch, (extraction, batch_reports)) in enumerate(zip(batches, results, strict=True), start=1):
                extracted.extend(extraction.triples)
                reports.append({"batch": batch_no, "passages": len(batch), "edc": batch_reports})
            kept = [name for name in self.extraction.kept_relations if name not in state["inferred"].properties]
            if kept:
                # Relations kept from open extraction join the run's subschema (HARNESS_UNMAPPED_RELATIONS=keep).
                state["inferred"] = state["inferred"].model_copy(
                    update={"properties": [*state["inferred"].properties, *kept]})
                self._emit("Keeping unmapped relations", f"{len(kept)} new relations: {', '.join(kept[:6])}"
                                                         + (" …" if len(kept) > 6 else ""))
        elif self.enhanced:
            extracted, reports = self.extraction.extract_batches(
                self._research_question(state), state["inferred"], state["passages"]
            )
        else:
            extracted = self.extraction.extract(
                self._research_question(state), state["inferred"], state["passages"]
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
        if not extracted:
            raise NoResults("extraction contained no valid supported propositions")
        # The evidence graph: every extracted triple, with the evidence review's verdict when there is one.
        # The answer uses only the relevant ones; this keeps the rest (evidence_graph.json).
        evidence = [{**item.model_dump(), "relevant": None} for item in extracted]
        if self.refine:
            extracted, review = self.extraction.refine_triples(
                self._research_question(state), extracted, state["passages"], reviewer=self.refinement_model
            )
            evidence = review.pop("labelled", evidence)
            reports.append({"stage": "evidence_review", **review})
            if not extracted:
                raise NoResults("evidence review kept no extracted triple as relevant to the question")
        self._emit("Extracting triples", f"{len(extracted)} supported triples")
        self._check("extraction", "pass", f"{len(extracted)} supported triples")
        return {"extracted": extracted, "inferred": state["inferred"], "extraction_batches": reports,
                "evidence_graph": evidence}

    def _facts_answer(self, triples: list[ExtractedTriple], removed: list[dict[str, Any]]) -> AnswerResult:
        """An answer that states the extracted facts, one sentence each (at most 12), with the
        composer's rejected sentences kept as ``removed`` so the reader sees what was dropped."""
        sentences: list[dict[str, Any]] = [
            {**item, "verdict": "removed"} for item in removed if item.get("verdict") == "removed"]
        for triple in triples[:12]:
            title = self.corpus.paper_title(triple.subject) or triple.subject
            relation = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", triple.predicate).lower()
            sentences.append({"text": f"{title}: {relation} {triple.object} [{triple.source_id}].",
                              "sourceIds": [triple.source_id], "verdict": "from_facts",
                              "reason": "stated from an extracted fact"})
        stated = [item for item in sentences if item["verdict"] == "from_facts"]
        cited = list(dict.fromkeys(source for item in stated for source in item["sourceIds"]))
        return AnswerResult(answer=" ".join(item["text"] for item in stated), cited_source_ids=cited,
                            sparql=self.extraction.answer_query(triples, cited), sentences=sentences)

    def _query_answer(self, state: BaselineState) -> dict[str, Any]:
        self.stage = "answer composition"
        notes = self._research_notes(state)
        report: dict[str, Any] = {}
        try:
            answer = self.extraction.answer(
                self._research_question(state), state["inferred"], state["extracted"],
                verify=self.refine, verifier=self.refinement_model, report=report, **({"notes": notes} if notes else {}),
            )
        except UnsupportedAnswer as exc:
            # After one repair nothing the model wrote was supported, but evidence review kept these
            # triples: state them instead of failing the run (they are supported by construction).
            # Only when evidence review kept the triples (Adaptive/Agentic): then they are supported by
            # construction. Without review (Fast) the run still fails as answer_unsupported.
            if exc.stage not in {"verification", "citation"} or not state["extracted"] or not self.refine:
                raise
            answer = self._facts_answer(state["extracted"], exc.sentences)
            after = " after one repair" if report.get("repaired") else ""
            why = (f"no composed sentence cited a source of the triples{after} (it cited "
                   f"{', '.join(exc.unknown_sources[:3])})" if exc.stage == "citation"
                   else f"no composed sentence was supported{after} "
                        f"({len(exc.sentences) or len(exc.removed)} removed)")
            self._check("answer composition", "warn", f"{why}; the answer states "
                        f"{min(12, len(state['extracted']))} of the {len(state['extracted'])} extracted facts")
            report = {}
        if report.get("citations_dropped"):
            dropped = report["citations_dropped"]
            self._check("answer composition", "warn", f"{dropped['sentences']} sentence(s) citing ids that are no "
                        f"triple's source were dropped: {', '.join(dropped['ids'][:3])}")
        if report.get("batches"):
            batches = report["batches"]
            self._check("answer composition", "warn" if batches["unsupported"] else "info",
                        f"{batches['triples']} triples did not fit one prompt: answered in {batches['composed']} "
                        f"batches" + (f" (of {batches['total']}; HARNESS_ANSWER_MAX_BATCHES)" if
                                      batches["composed"] < batches["total"] else "")
                        + (f", {batches['unsupported']} without a supported answer left out"
                           if batches["unsupported"] else "") + "; every triple stays in the answer graph")
        if report.get("repaired"):
            self._check("answer composition", "warn", "the first answer had no supported, cited claim; "
                        "it was written once more from the triples"
                        + (" without the research notes" if report.get("notes_dropped") else ""))
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

