"""Triple extraction: direct (subschema predicates) and EDC (extract, define, canonicalize).

A mixin of ``ExtractionEngine``.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from harness_config import settings
from harness_prompts import render
from harness_semantic import HRN, TOP_CLASS
from rdflib import RDFS, URIRef

from .call_control import parallel_map
from .events import labelled
from .model_client import complete
from .naming import _literal_value, local_name, prompt_shape, relation_name
from .schemas import (
    EDCDefinition,
    EDCDefinitionResult,
    EDCOpenTriple,
    ExtractedTriple,
    ExtractionResult,
    SchemaInference,
)


class ExtractionStage:
    @labelled("triple extraction")
    def extract(
        self, question: str, inferred: SchemaInference, passages: list[dict[str, Any]]
    ) -> ExtractionResult:
        result = ExtractionResult.model_validate(complete(self.model, render(
            "extraction/triple_extraction",
            {
                "question": question,
                "classes": inferred.classes,
                "properties": inferred.properties,
                "property_signatures": self.signatures(list(inferred.properties)),
                "passages": [
                    {
                        "passage_id": row["passage_id"],
                        "paper_id": row["paper_id"],
                        "text": row["text"][:1800],
                    }
                    for row in passages
                ],
            }, **prompt_shape(),
        )))
        passage_ids = {row["passage_id"] for row in passages}
        # Models sometimes return a useful supported triple alongside a
        # natural-language relation or an invalid passage id. Keep the valid
        # evidence instead of rejecting the whole extraction response.
        result.triples = [
            triple
            for triple in result.triples
            if triple.predicate in inferred.properties and triple.source_id in passage_ids
        ]
        return result

    @labelled("open extraction and relation mapping")
    def extract_edc(
        self, question: str, inferred: SchemaInference, passages: list[dict[str, Any]]
    ) -> tuple[ExtractionResult, list[dict[str, Any]]]:
        """Run EDC-style Extract -> Define -> Canonicalize over passages.

        Extraction keeps predicates open and ties every proposition to evidence.
        Canonicalization then maps only high-confidence open predicates to the
        inferred ontology; unsupported relations remain unresolved instead of
        being forced into a schema property.
        """
        paper_by_source = {str(row["passage_id"]): str(row["paper_id"]) for row in passages}
        open_triples = self._open_triples(question, passages)
        definitions = self._define(open_triples)
        definition_map = {self._norm(item.relation): item.definition for item in definitions}
        relations = list(inferred.properties)
        selected_relations = [
            self._canonical_property(item.predicate, definition_map.get(self._norm(item.predicate), ""), relations)
            for item in open_triples
        ]
        unresolved = [
            {"id": index, "subject": item.subject, "predicate": item.predicate,
             "object": item.object, "evidence": item.evidence}
            for index, (item, selection) in enumerate(zip(open_triples, selected_relations, strict=True))
            if selection[0] is None
        ]
        llm_mapping = self._map_by_model(question, unresolved, relations, len(open_triples)) \
            if unresolved and relations else {}
        canonical, decisions, unmapped = self._canonicalize(open_triples, selected_relations, llm_mapping,
                                                            definition_map, paper_by_source)
        reports: list[dict[str, Any]] = [
            {"stage": "extract", "open_triples": len(open_triples)},
            {"stage": "define", "definitions": len(definitions)},
            {"stage": "ontology_review", "unresolved": len(unresolved), "mapped": len(llm_mapping)},
            {"stage": "canonicalize", "decisions": decisions},
        ]
        if unmapped:
            # Facts lost because their relation fits no subschema term (HARNESS_UNMAPPED_RELATIONS=drop).
            reports.append({"stage": "unmapped_dropped", "count": len(unmapped), "triples": unmapped})
        return ExtractionResult(triples=canonical), reports

    def _open_triples(self, question: str, passages: list[dict[str, Any]]) -> list[EDCOpenTriple]:
        """Extract: open propositions, each tied to at least one of the given passages."""
        allowed_ids = {row["passage_id"] for row in passages}
        payload = complete(self.edc_model, render("extraction/edc_extract", {
            "question": question,
            "passages": [{"source_id": row["passage_id"], "paper_id": row["paper_id"],
                          "text": row["text"][:1800]} for row in passages],
        }, **prompt_shape()))
        raw = payload.get("triples", []) if isinstance(payload, dict) else []
        open_triples: list[EDCOpenTriple] = []
        for item in raw if isinstance(raw, list) else []:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source_id") or "").strip() or None
            ids = [str(x) for x in item.get("evidence_ids", []) if str(x) in allowed_ids]
            if source in allowed_ids and source not in ids:
                ids.append(source)
            if not ids:
                continue
            try:
                open_triples.append(EDCOpenTriple.model_validate({
                    **item,
                    "source_id": source,
                    "evidence_ids": ids,
                    "evidence": item.get("evidence") or item.get("evidence_span") or "",
                }))
            except (TypeError, ValueError):
                continue
        return open_triples

    def _define(self, open_triples: list[EDCOpenTriple]) -> list[EDCDefinition]:
        """Define: one definition per distinct open relation."""
        distinct = sorted({item.predicate.strip() for item in open_triples if item.predicate.strip()})
        if not distinct:
            return []
        defined = complete(self.edc_model, render("extraction/edc_define", {"relations": distinct}))
        try:
            return EDCDefinitionResult.model_validate(defined).definitions
        except (TypeError, ValueError):
            return []

    def _property_catalog(self, relations: list[str]) -> list[dict[str, Any]]:
        catalog = []
        for relation in relations:
            iri = self.property_names.get(relation)
            domain = self.schema.graph.value(iri, RDFS.domain) if iri else None
            range_ = self.schema.graph.value(iri, RDFS.range) if iri else None
            entry = {
                "property": relation,
                "domain": local_name(domain) if isinstance(domain, URIRef) else None,
                "range": local_name(range_) if isinstance(range_, URIRef) else None,
            }
            if iri is not None:
                # The model tells paper-level from entity-level relations (addresses vs solves)
                # by their definitions too.
                entry["definition"] = " ".join(str(self.schema.graph.value(iri, RDFS.comment) or "").split())
            catalog.append(entry)
        return catalog

    def _map_by_model(self, question: str, unresolved: list[dict[str, Any]], relations: list[str],
                      count: int) -> dict[int, tuple[str, float]]:
        """Canonicalize, second chance: the mapping model's confident choices for unresolved relations."""
        mapped = complete(self.edc_model, render(
            "extraction/edc_map",
            {"question": question, "unresolved": unresolved,
             "allowed_properties": self._property_catalog(relations)},
            ensure_ascii=False,
        ))
        mapping: dict[int, tuple[str, float]] = {}
        for row in mapped.get("mappings", []) if isinstance(mapped, dict) else []:
            if not isinstance(row, dict):
                continue
            index = row.get("id")
            property_name = row.get("property")
            try:
                confidence = float(row.get("confidence", 0))
            except (TypeError, ValueError):
                continue
            if (type(index) is int and 0 <= index < count and property_name in relations
                    and confidence >= settings().map_model_min):
                mapping[index] = (property_name, confidence)
        return mapping

    def _canonicalize(self, open_triples: list[EDCOpenTriple], selected_relations: list[tuple[str | None, float]],
                      llm_mapping: dict[int, tuple[str, float]], definition_map: dict[str, str],
                      paper_by_source: dict[str, str]
                      ) -> tuple[list[ExtractedTriple], list[dict[str, Any]], list[dict[str, str]]]:
        """Each open fact under its chosen property (lexical, then model, then kept as a new relation);
        returns the triples, one decision per open fact, and the facts left unmapped."""
        canonical: list[ExtractedTriple] = []
        decisions: list[dict[str, Any]] = []
        keep = settings().unmapped_relations == "keep"
        papers = set(paper_by_source.values())
        unmapped: list[dict[str, str]] = []
        for index, item in enumerate(open_triples):
            selected, score = selected_relations[index]
            method = "lexical"
            if selected is None and index in llm_mapping:
                selected, score = llm_mapping[index]
                method = "ontology_review"
            if selected is None and keep:
                selected = self._keep_relation(item, definition_map.get(self._norm(item.predicate), ""), papers)
                method = "kept_new_relation" if selected else method
            decisions.append({"open": item.predicate, "selected": selected, "score": round(score, 3), "method": method})
            if selected is None:
                unmapped.append({"subject": item.subject, "predicate": item.predicate, "object": item.object,
                                 "source_id": item.evidence_ids[0]})
                continue
            property_iri = self.property_names.get(selected)
            domain = self.schema.graph.value(property_iri, RDFS.domain) if property_iri else None
            subject = paper_by_source.get(item.evidence_ids[0], item.subject) if domain == HRN.Paper else item.subject
            canonical.append(ExtractedTriple(
                subject=subject,
                predicate=selected,
                object=item.object,
                source_id=item.evidence_ids[0],
                evidence=item.evidence or f"Evidence from {item.evidence_ids[0]}.",
            ))
        return canonical, decisions, unmapped

    def _keep_relation(self, item: EDCOpenTriple, definition: str, papers: set[str]) -> str | None:
        """Register the open fact's own relation (HARNESS_UNMAPPED_RELATIONS=keep) and return its name.

        The relation is an OWL 2 sub-property of the generic researchRelation (object range) or
        researchAttribute (a value), so the graph stays valid and the scorer's entailed level can
        relate it to other terms. Its domain is Paper when the subject is the passage's paper,
        else ResearchEntity.
        """
        name = relation_name(item.predicate)
        if name is None:
            return None
        if name in self.property_names:
            return name
        about_paper = item.subject in papers
        try:
            self.register_local_relation(
                name,
                definition or f"Relation kept from open extraction, worded '{item.predicate}'.",
                domain_name="Paper" if about_paper else local_name(TOP_CLASS),
                range_name="xsd:string" if _literal_value(item.object) else local_name(TOP_CLASS),
            )
        except ValueError:
            return None
        with self._schema_lock:
            self.kept_relations.setdefault(name, item.predicate)
        return name

    @staticmethod
    def _norm(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()

    @classmethod
    def _canonical_property(cls, predicate: str, definition: str, relations: list[str]) -> tuple[str | None, float]:
        source = set(cls._norm(f"{predicate} {definition}").split())
        ranked = []
        for relation in relations:
            target = set(cls._norm(relation).split())
            overlap = len(source & target) / max(1, len(source | target))
            ratio = SequenceMatcher(None, cls._norm(predicate), cls._norm(relation)).ratio()
            ranked.append((0.55 * ratio + 0.45 * overlap, relation))
        if not ranked:
            return None, 0.0
        score, relation = max(ranked)
        return (relation, score) if score >= settings().map_lexical_min else (None, score)

    def extract_batches(
        self,
        question: str,
        inferred: SchemaInference,
        passages: list[dict[str, Any]],
        batch_size: int = 4,
        text_limit: int = 2500,
    ) -> tuple[list[ExtractedTriple], list[dict[str, Any]]]:
        """Extract from bounded passage batches and deduplicate with provenance."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        merged: dict[tuple[str, str, str], ExtractedTriple] = {}
        reports: list[dict[str, Any]] = []
        batches = [passages[start : start + batch_size] for start in range(0, len(passages), batch_size)]

        def run_batch(batch: list[dict[str, Any]]) -> ExtractionResult | ValueError:
            try:
                return self.extract(question, inferred, [{**row, "text": row["text"][:text_limit]} for row in batch])
            except ValueError as exc:
                return exc

        outcomes = parallel_map(run_batch, batches, settings().extraction_concurrency)
        for batch_no, (batch, result) in enumerate(zip(batches, outcomes, strict=True), start=1):
            if isinstance(result, ValueError):
                reports.append({"batch": batch_no, "passages": [p["passage_id"] for p in batch], "error": str(result)})
                continue
            reports.append({"batch": batch_no, "passages": [p["passage_id"] for p in batch], "triples": len(result.triples)})
            for triple in result.triples:
                key = (triple.subject.casefold().strip(), triple.predicate, triple.object.casefold().strip())
                previous = merged.get(key)
                if previous is None:
                    merged[key] = triple
                elif triple.source_id != previous.source_id:
                    merged[key] = previous.model_copy(update={"evidence": f"{previous.evidence} | {triple.evidence}"})
        return list(merged.values()), reports
