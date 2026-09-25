"""The extraction engine: schema inference, triple extraction, review and answers.

One interface shared by the baseline paths and the agentic path. Every model
call renders a named prompt from ``harness_prompts`` (see the prompt name at
each call) and goes through the ``ModelClient`` protocol.
"""

from __future__ import annotations

import re
from dataclasses import replace
from difflib import SequenceMatcher
from typing import Any

from harness_prompts import render
from harness_semantic import HRN, Subschema
from rdflib import RDF, RDFS, XSD, Literal, URIRef

from .errors import ModelFailure
from .events import emit_model_event, labelled
from .model_client import ModelClient, complete
from .schemas import (
    AnswerResult,
    EDCDefinition,
    EDCDefinitionResult,
    EDCOpenTriple,
    ExtractedTriple,
    ExtractionResult,
    SchemaInference,
)


def local_name(value: URIRef) -> str:
    return str(value).rsplit("#", 1)[-1]


class ExtractionEngine:
    """One interface shared by baseline and the later agentic path."""

    def __init__(
        self,
        model: ModelClient,
        schema: Subschema,
        schema_model: ModelClient | None = None,
        edc_model: ModelClient | None = None,
    ) -> None:
        self.model = model
        self.schema_model = schema_model or model
        self.edc_model = edc_model or model
        self.schema = schema
        self.class_names = {local_name(iri): iri for iri in schema.classes}
        self.property_names = {local_name(iri): iri for iri in schema.properties}
        self._local_schema = False

    def _ensure_local_schema(self) -> None:
        if self._local_schema:
            return
        graph = self.schema.graph.__class__()
        for prefix, namespace in self.schema.graph.namespaces():
            graph.bind(prefix, namespace)
        for triple in self.schema.graph:
            graph.add(triple)
        self.schema = replace(self.schema, graph=graph)
        self._local_schema = True

    def register_local_class(self, name: str, definition: str) -> URIRef:
        """Introduce one query-local class without changing the seed ontology."""
        if not re.fullmatch(r"[A-Z][A-Za-z0-9_]{0,63}", name):
            raise ValueError(f"invalid proposed class name: {name!r}")
        if name in self.class_names:
            return self.class_names[name]
        self._ensure_local_schema()
        iri = URIRef(HRN[name])
        self.schema.graph.add((iri, RDF.type, RDFS.Class))
        self.schema.graph.add((iri, RDFS.label, Literal(name)))
        self.schema.graph.add((iri, RDFS.comment, Literal(definition)))
        self.schema = replace(self.schema, classes=self.schema.classes | {iri})
        self.class_names[name] = iri
        return iri

    def register_local_relation(
        self, name: str, definition: str, *, domain_name: str = "Paper",
        range_name: str = "xsd:string",
    ) -> URIRef:
        """Add a query-local relation with an explicit domain and range."""
        if not re.fullmatch(r"[a-z][A-Za-z0-9_]{0,63}", name):
            raise ValueError(f"invalid proposed relation name: {name!r}")
        iri = URIRef(HRN[name])
        if iri in self.schema.properties:
            return iri
        self._ensure_local_schema()
        graph = self.schema.graph
        graph.add((iri, RDF.type, RDF.Property))
        graph.add((iri, RDFS.label, Literal(name)))
        graph.add((iri, RDFS.comment, Literal(definition)))
        graph.add((iri, RDFS.domain, self.class_names.get(domain_name, HRN.Paper)))
        graph.add((iri, RDFS.range, self.class_names.get(range_name, XSD.string)))
        self.schema = replace(self.schema, properties=self.schema.properties | {iri})
        self.property_names[name] = iri
        return iri

    @labelled("schema inference")
    def infer_schema(self, question: str, passages: list[dict[str, Any]],
                     notes: list[str] | None = None) -> SchemaInference:
        """``notes``: the agent's unverified findings (agentic runs), sent as their own field so they
        never become part of a definition or the question text."""
        allowed_classes = sorted(self.class_names)
        allowed_properties = sorted(self.property_names)
        payload: dict[str, Any] = {
            "question": question,
            "allowed_classes": allowed_classes,
            "allowed_properties": allowed_properties,
            "passages": [
                {"source_id": row["passage_id"], "text": row["text"][:1200]}
                for row in passages
            ],
        }
        if notes:
            payload["research_notes"] = list(notes)
        result = SchemaInference.model_validate(complete(self.schema_model, render(
            "extraction/schema_inference", payload,
        )))
        if not result.classes or not result.properties:
            raise ValueError("schema inference returned no classes or properties")
        class_defs = {
            item["name"]: item for item in result.class_definitions
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        }
        property_defs = {
            item["name"]: item for item in result.property_definitions
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        }
        for name in result.classes:
            if name not in self.class_names:
                definition = str(class_defs.get(name, {}).get("definition") or
                                 f"Run-local research concept {name} for: {question}")
                self.register_local_class(name, definition)
        for name in result.properties:
            if name in self.property_names:
                continue
            spec = property_defs.get(name, {})
            range_name = str(spec.get("range") or "")
            if range_name not in self.class_names:
                range_name = next(
                    (class_name for class_name in sorted(result.classes, key=len, reverse=True)
                     if name.casefold().endswith(class_name.casefold())),
                    "xsd:string",
                )
            definition = str(spec.get("definition") or
                             f"Run-local research relation {name} for: {question}")
            self.register_local_relation(
                name, definition, domain_name=str(spec.get("domain") or "Paper"),
                range_name=range_name,
            )
        new_classes = set(result.classes) - set(allowed_classes)
        new_properties = set(result.properties) - set(allowed_properties)
        if new_classes or new_properties:
            emit_model_event(
                "Schema extension proposed",
                f"classes: {', '.join(sorted(new_classes)) or 'none'}; "
                f"properties: {', '.join(sorted(new_properties)) or 'none'}",
            )
        return result

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
                "passages": [
                    {
                        "passage_id": row["passage_id"],
                        "paper_id": row["paper_id"],
                        "text": row["text"][:1800],
                    }
                    for row in passages
                ],
            },
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
        allowed_ids = {row["passage_id"] for row in passages}
        paper_by_source = {str(row["passage_id"]): str(row["paper_id"]) for row in passages}
        payload = complete(self.edc_model, render("extraction/edc_extract", {
            "question": question,
            "passages": [{"source_id": row["passage_id"], "paper_id": row["paper_id"],
                          "text": row["text"][:1800]} for row in passages],
        }))
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

        relations = list(inferred.properties)
        definitions: list[EDCDefinition] = []
        distinct = sorted({item.predicate.strip() for item in open_triples if item.predicate.strip()})
        if distinct:
            defined = complete(self.edc_model, render("extraction/edc_define", {"relations": distinct}))
            try:
                definitions = EDCDefinitionResult.model_validate(defined).definitions
            except (TypeError, ValueError):
                definitions = []
        definition_map = {self._norm(item.relation): item.definition for item in definitions}
        selected_relations = [
            self._canonical_property(
                item.predicate, definition_map.get(self._norm(item.predicate), ""), relations
            )
            for item in open_triples
        ]
        unresolved = [
            {"id": index, "subject": item.subject, "predicate": item.predicate,
             "object": item.object, "evidence": item.evidence}
            for index, (item, selection) in enumerate(zip(open_triples, selected_relations, strict=True))
            if selection[0] is None
        ]
        llm_mapping: dict[int, tuple[str, float]] = {}
        if unresolved and relations:
            property_catalog = []
            for relation in relations:
                iri = self.property_names.get(relation)
                domain = self.schema.graph.value(iri, RDFS.domain) if iri else None
                range_ = self.schema.graph.value(iri, RDFS.range) if iri else None
                property_catalog.append({
                    "property": relation,
                    "domain": local_name(domain) if isinstance(domain, URIRef) else None,
                    "range": local_name(range_) if isinstance(range_, URIRef) else None,
                })
            mapped = complete(self.edc_model, render(
                "extraction/edc_map",
                {"question": question, "unresolved": unresolved,
                 "allowed_properties": property_catalog},
                ensure_ascii=False,
            ))
            for row in mapped.get("mappings", []) if isinstance(mapped, dict) else []:
                if not isinstance(row, dict):
                    continue
                index = row.get("id")
                property_name = row.get("property")
                try:
                    confidence = float(row.get("confidence", 0))
                except (TypeError, ValueError):
                    continue
                if type(index) is int and 0 <= index < len(open_triples) and property_name in relations and confidence >= 0.7:
                    llm_mapping[index] = (property_name, confidence)

        canonical: list[ExtractedTriple] = []
        decisions: list[dict[str, Any]] = []
        for index, item in enumerate(open_triples):
            selected, score = selected_relations[index]
            method = "lexical"
            if selected is None and index in llm_mapping:
                selected, score = llm_mapping[index]
                method = "ontology_review"
            decisions.append({"open": item.predicate, "selected": selected, "score": round(score, 3), "method": method})
            if selected is None:
                continue
            property_iri = self.property_names.get(selected)
            domain = self.schema.graph.value(property_iri, RDFS.domain) if property_iri else None
            subject = (
                paper_by_source.get(item.evidence_ids[0], item.subject)
                if domain == HRN.Paper
                else item.subject
            )
            canonical.append(ExtractedTriple(
                subject=subject,
                predicate=selected,
                object=item.object,
                source_id=item.evidence_ids[0],
                evidence=item.evidence or f"Evidence from {item.evidence_ids[0]}.",
            ))
        return ExtractionResult(triples=canonical), [
            {"stage": "extract", "open_triples": len(open_triples)},
            {"stage": "define", "definitions": len(definitions)},
            {"stage": "ontology_review", "unresolved": len(unresolved), "mapped": len(llm_mapping)},
            {"stage": "canonicalize", "decisions": decisions},
        ]

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
        return (relation, score) if score >= 0.42 else (None, score)

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
        for batch_no, start in enumerate(range(0, len(passages), batch_size), start=1):
            batch = passages[start : start + batch_size]
            try:
                result = self.extract(question, inferred, [
                    {**row, "text": row["text"][:text_limit]} for row in batch
                ])
            except ValueError as exc:
                reports.append({"batch": batch_no, "passages": [p["passage_id"] for p in batch], "error": str(exc)})
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

    @labelled("evidence review")
    def refine_triples(
        self,
        question: str,
        triples: list[ExtractedTriple],
        passages: list[dict[str, Any]],
        batch_size: int = 24,
        reviewer: ModelClient | None = None,
    ) -> tuple[list[ExtractedTriple], dict[str, int]]:
        """Second model pass: retain only atomic claims relevant to the question.

        The reviewer receives numbered existing candidates and may only return
        their IDs. It cannot add a fact or change provenance. Batches cover the
        entire candidate set, so adaptive retrieval has no output cap here.
        """
        passage_text = {str(row["passage_id"]): str(row["text"]) for row in passages}
        deduped: dict[tuple[str, str, str], ExtractedTriple] = {}
        for item in triples:
            if not all((item.subject.strip(), item.predicate.strip(), item.object.strip(), item.source_id.strip())):
                continue
            key = (item.subject.casefold().strip(), item.predicate, item.object.casefold().strip())
            deduped.setdefault(key, item)
        candidates = list(deduped.values())
        kept: list[ExtractedTriple] = []
        calls = 0
        for start in range(0, len(candidates), batch_size):
            batch = candidates[start : start + batch_size]
            payload = {
                "question": question,
                "candidates": [
                    {
                        "id": i,
                        **item.model_dump(),
                        "source_text": passage_text.get(item.source_id, "")[:1500],
                    }
                    for i, item in enumerate(batch)
                ],
            }
            decision = complete(reviewer or self.model,
                                render("extraction/evidence_review", payload, ensure_ascii=False))
            calls += 1
            selected = decision.get("keep_ids", []) if isinstance(decision, dict) else []
            if not isinstance(selected, list):
                raise TypeError("triple reviewer must return keep_ids as a list")
            indices = {value for value in selected if type(value) is int and 0 <= value < len(batch)}
            kept.extend(item for i, item in enumerate(batch) if i in indices)
        return kept, {"candidates": len(triples), "unique": len(candidates), "kept": len(kept), "model_calls": calls}

    @labelled("answer composition and verification")
    def answer(
        self,
        question: str,
        inferred: SchemaInference,
        triples: list[ExtractedTriple],
        verify: bool = False,
        verifier: ModelClient | None = None,
        notes: list[str] | None = None,
    ) -> AnswerResult:
        """``notes`` (agentic runs) go to composition only; the verifier sees the plain question."""
        payload: dict[str, Any] = {
            "question": question,
            "classes": inferred.classes,
            "properties": inferred.properties,
            "triples": [triple.model_dump() for triple in triples],
        }
        if notes:
            payload["research_notes"] = list(notes)
        try:
            reply = complete(self.model, render("extraction/answer_composition", payload))
        except ModelFailure as exc:
            if "output limit" not in str(exc):
                raise
            # Cut off (a very long answer or a repetition loop): once more with explicit limits.
            emit_model_event("Answer too long", "retrying answer composition with a shorter format")
            reply = complete(self.model, render("extraction/answer_composition", payload, brief=True))
        result = AnswerResult.model_validate(reply)
        sources = {triple.source_id for triple in triples}
        if not set(result.cited_source_ids) <= sources:
            raise ValueError("answer cites a source outside the extracted evidence")
        if verify:
            reviewed = complete(verifier or self.model, render(
                "extraction/answer_verification",
                {
                    "question": question,
                    "draft_answer": result.answer,
                    "triples": [triple.model_dump() for triple in triples],
                },
                ensure_ascii=False,
            ))
            if reviewed.get("supported") is not True or not str(reviewed.get("answer", "")).strip():
                raise ValueError("answer has no fully supported factual claim")
            verified_sources = reviewed.get("cited_source_ids", [])
            if not isinstance(verified_sources, list) or not verified_sources or not set(verified_sources) <= sources:
                raise ValueError("verified answer lacks valid triple provenance")
            result = result.model_copy(update={
                "answer": str(reviewed["answer"]),
                "cited_source_ids": [str(source) for source in verified_sources],
            })
        return result

    def axioms(self, inferred: SchemaInference) -> list[tuple[str, str, str]]:
        rows = []
        for name in inferred.properties:
            iri = self.property_names[name]
            domain = self.schema.graph.value(iri, RDFS.domain)
            range_ = self.schema.graph.value(iri, RDFS.range)
            if isinstance(domain, URIRef) and isinstance(range_, URIRef):
                range_name = (
                    f"xsd:{local_name(range_)}"
                    if str(range_).startswith("http://www.w3.org/2001/XMLSchema#")
                    else local_name(range_)
                )
                rows.append((name, local_name(domain), range_name))
        return rows


__all__ = ["ExtractionEngine", "local_name"]
