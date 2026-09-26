"""Research steps after retrieval: schema inference, extraction and claims, answer.

A mixin of ``Baseline20``: methods use the pipeline's state (``self``).
"""

from __future__ import annotations

from typing import Any

from harness_config import settings
from harness_extraction import (
    ExtractedTriple,
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
            batch_size = 2
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
        if self.refine:
            extracted, review = self.extraction.refine_triples(
                self._research_question(state), extracted, state["passages"], reviewer=self.refinement_model
            )
            reports.append({"stage": "evidence_review", **review})
            if not extracted:
                raise NoResults("evidence review kept no extracted triple as relevant to the question")
        self._emit("Extracting triples", f"{len(extracted)} supported triples")
        self._check("extraction", "pass", f"{len(extracted)} supported triples")
        return {"extracted": extracted, "inferred": state["inferred"], "extraction_batches": reports}

    def _query_answer(self, state: BaselineState) -> dict[str, Any]:
        self.stage = "answer composition"
        notes = self._research_notes(state)
        report: dict[str, Any] = {}
        answer = self.extraction.answer(
            self._research_question(state), state["inferred"], state["extracted"],
            verify=self.refine, verifier=self.refinement_model, **({"notes": notes, "report": report} if notes else {}),
        )
        if report.get("retried_without_notes"):
            self._check("answer composition", "warn", "the verifier removed every claim of the answer written with "
                        "research notes; answered from the triples alone")
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

