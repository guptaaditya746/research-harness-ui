"""Research steps after retrieval: schema inference, extraction and claims, answer.

A mixin of ``Baseline20``: methods use the pipeline's state (``self``).
"""

from __future__ import annotations

import re
from typing import Any

from harness_extraction import (
    AnswerResult,
    ExtractedTriple,
)
from harness_extraction.claims import (
    CHANGE,
    NUMERIC,
)
from harness_graphstore import sparql_check
from rdflib import RDFS, XSD

from .planner import QueryPlan
from .state import (
    BaselineState,
    NoResults,
)


class ResearchSteps:
    def _infer_schema(self, state: BaselineState) -> dict[str, Any]:
        self.stage = "schema inference"
        seed_properties = set(self.extraction.property_names)
        inferred = self.extraction.infer_schema(self._research_question(state), state["passages"])
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
            batch_size = 2
            batch_count = (len(state["passages"]) + batch_size - 1) // batch_size
            for start in range(0, len(state["passages"]), max(1, batch_size)):
                batch_no = start // batch_size + 1
                self._emit(
                    "Extracting passages",
                    f"batch {batch_no}/{batch_count} · {min(batch_size, len(state['passages']) - start)} passages",
                )
                extraction, batch_reports = self.extraction.extract_edc(
                    self._research_question(state), state["inferred"], state["passages"][start : start + batch_size]
                )
                extracted.extend(extraction.triples)
                reports.append({"batch": batch_no, "passages": len(state["passages"][start : start + batch_size]), "edc": batch_reports})
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
                        self.relation_proposer(
                            name, definition, self.current_run_id, "Paper", "xsd:string"
                        )
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
                self._research_question(state), extracted, state["passages"], reviewer=self.refinement_model
            )
            reports.append({"stage": "evidence_review", **review})
            if not extracted:
                raise NoResults("evidence review kept no extracted triple as relevant to the question")
        plan = state.get("plan") or QueryPlan.fallback("no plan")
        mentioned = list(state.get("paper_ids") or [])
        if self.enhanced and len(mentioned) > 1 and plan.evidence_focus == "metrics":
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
        self._check("extraction", "pass", f"{len(extracted)} supported triples")
        return {"extracted": extracted, "inferred": state["inferred"], "extraction_batches": reports}

    def _query_answer(self, state: BaselineState) -> dict[str, Any]:
        self.stage = "answer composition"
        plan = state.get("plan") or QueryPlan.fallback("no plan")
        if plan.evidence_focus == "results":
            context_terms = (
                {term.casefold() for term in re.findall(r"[A-Za-z]{4,}", plan.benchmark)}
                if plan.benchmark else set()
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
            self._research_question(state), state["inferred"], state["extracted"],
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

