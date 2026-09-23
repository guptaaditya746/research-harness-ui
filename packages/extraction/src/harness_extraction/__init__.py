"""Shared structured model calls for schema inference, extraction, and answers."""

from __future__ import annotations

import json
import os
import re
import time
from difflib import SequenceMatcher
from typing import Any, Protocol

import httpx
from harness_semantic import HRN, Subschema
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator
from rdflib import RDFS, URIRef


class ModelFailure(RuntimeError):
    """Raised after the configured provider retries are exhausted."""


class ModelClient(Protocol):
    def complete_json(self, system: str, user: str) -> dict[str, Any]: ...


def _json_object(text: str) -> dict[str, Any]:
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1) if fenced else text[text.find("{") : text.rfind("}") + 1]
    value = json.loads(candidate)
    if not isinstance(value, dict):
        raise TypeError("model response must be a JSON object")
    return value


class OpenAICompatibleModel:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout_s: float = 30,
        retries: int | None = None,
    ) -> None:
        self.base_url = (
            base_url or os.getenv("LITELLM_BASE_URL", "http://127.0.0.1:4000/v1")
        ).rstrip("/")
        self.model = model or os.getenv("HARNESS_MODEL_SMALL", "lite-model")
        self.api_key = api_key or os.getenv("VLLM_API_KEY", "local")
        self.timeout_s = timeout_s
        self.retries = retries or int(os.getenv("PROVIDER_MAX_RETRIES", "3"))

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        error: Exception | None = None
        for attempt in range(self.retries):
            try:
                with httpx.Client(timeout=self.timeout_s) as client:
                    response = client.post(
                        f"{self.base_url}/chat/completions",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={
                            "model": self.model,
                            "temperature": 0,
                            "response_format": {"type": "json_object"},
                            "messages": [
                                {"role": "system", "content": system},
                                {"role": "user", "content": user},
                            ],
                        },
                    )
                    response.raise_for_status()
                    return _json_object(response.json()["choices"][0]["message"]["content"])
            except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                error = exc
                if attempt + 1 < self.retries:
                    time.sleep(0.25 * (2**attempt))
        raise ModelFailure(f"provider failed after {self.retries} attempts: {error}")


class SchemaInference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str
    gloss: str
    classes: list[str]
    properties: list[str]

    @model_validator(mode="before")
    @classmethod
    def normalize_catalogs(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        for key in ("classes", "properties"):
            item = normalized.get(key)
            if isinstance(item, dict):
                normalized[key] = list(item)
        return normalized


class ExtractedTriple(BaseModel):
    model_config = ConfigDict(extra="forbid")
    subject: str
    predicate: str
    object: str
    source_id: str
    evidence: str = Field(validation_alias=AliasChoices("evidence", "evidence_span"))


class ExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    triples: list[ExtractedTriple] = Field(default_factory=list)


class EDCOpenTriple(BaseModel):
    """Evidence-linked open proposition produced by EDC Extract."""

    model_config = ConfigDict(extra="ignore")
    subject: str
    predicate: str
    object: str
    source_id: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    evidence: str = ""


class EDCDefinition(BaseModel):
    relation: str
    definition: str


class EDCDefinitionResult(BaseModel):
    definitions: list[EDCDefinition] = Field(default_factory=list)


class AnswerResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sparql: str
    answer: str
    cited_source_ids: list[str]


def local_name(value: URIRef) -> str:
    return str(value).rsplit("#", 1)[-1]


class ExtractionEngine:
    """One interface shared by baseline and the later agentic path."""

    def __init__(self, model: ModelClient, schema: Subschema) -> None:
        self.model = model
        self.schema = schema
        self.class_names = {local_name(iri): iri for iri in schema.classes}
        self.property_names = {local_name(iri): iri for iri in schema.properties}

    def infer_schema(self, question: str, passages: list[dict[str, Any]]) -> SchemaInference:
        result = SchemaInference.model_validate(
            self.model.complete_json(
                "Select only terms from the supplied ontology. Return strict JSON with label, gloss, classes, properties.",
                json.dumps(
                    {
                        "question": question,
                        "allowed_classes": sorted(self.class_names),
                        "allowed_properties": sorted(self.property_names),
                        "passages": [
                            {"source_id": row["passage_id"], "text": row["text"][:1200]}
                            for row in passages
                        ],
                    }
                ),
            )
        )
        unknown = set(result.classes) - self.class_names.keys()
        unknown |= set(result.properties) - self.property_names.keys()
        if unknown:
            raise ValueError(f"schema terms outside the hand-written subschema: {sorted(unknown)}")
        if not result.classes or not result.properties:
            raise ValueError("schema inference returned no classes or properties")
        return result

    def extract(
        self, question: str, inferred: SchemaInference, passages: list[dict[str, Any]]
    ) -> ExtractionResult:
        result = ExtractionResult.model_validate(
            self.model.complete_json(
                "Extract only supported triples. Use a supplied passage_id as source_id and copy a short evidence span. Return strict JSON with triples.",
                json.dumps(
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
                    }
                ),
            )
        )
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
        payload = self.model.complete_json(
            "Extract evidence-backed atomic propositions relevant to the question. Use the supplied paper_id as subject for paper-level claims. Keep predicates open natural-language relations. "
            "Return strict JSON with triples; each triple must include source_id (a supplied passage_id), subject, predicate, object, and evidence.",
            json.dumps({
                "question": question,
                "passages": [{"source_id": row["passage_id"], "paper_id": row["paper_id"], "text": row["text"][:1800]} for row in passages],
            }),
        )
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
            defined = self.model.complete_json(
                "Define each open predicate in subject-to-object terms. Return strict JSON with definitions, each containing relation and definition.",
                json.dumps({"relations": distinct}),
            )
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
            mapped = self.model.complete_json(
                "Map each open relation to one allowed ontology property only when the evidence supports the same subject-to-object claim. Return strict JSON with mappings: [{id, property, confidence}]. Use null property when none fits. Confidence must be between 0 and 1. Never add a fact or source.",
                json.dumps({"question": question, "unresolved": unresolved,
                            "allowed_properties": property_catalog}, ensure_ascii=False),
            )
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
            decision = (reviewer or self.model).complete_json(
                "You review candidate atomic knowledge-graph triples. Select only IDs whose exact subject-predicate-object claim answers or directly supports the question AND whose source text supports the claim. Never add triples. Return strict JSON {\"keep_ids\": [integer IDs]}. If none qualify, return an empty list.",
                json.dumps(payload, ensure_ascii=False),
            )
            calls += 1
            selected = decision.get("keep_ids", []) if isinstance(decision, dict) else []
            if not isinstance(selected, list):
                raise TypeError("triple reviewer must return keep_ids as a list")
            indices = {value for value in selected if type(value) is int and 0 <= value < len(batch)}
            kept.extend(item for i, item in enumerate(batch) if i in indices)
        return kept, {"candidates": len(triples), "unique": len(candidates), "kept": len(kept), "model_calls": calls}

    def answer(
        self,
        question: str,
        inferred: SchemaInference,
        triples: list[ExtractedTriple],
        verify: bool = False,
        verifier: ModelClient | None = None,
    ) -> AnswerResult:
        result = AnswerResult.model_validate(
            self.model.complete_json(
                "Write a concise answer supported only by the triples. Produce executable SPARQL over prefix hrn: <https://harness.local/schema#>. Return strict JSON with sparql, answer, cited_source_ids.",
                json.dumps(
                    {
                        "question": question,
                        "classes": inferred.classes,
                        "properties": inferred.properties,
                        "triples": [triple.model_dump() for triple in triples],
                    }
                ),
            )
        )
        sources = {triple.source_id for triple in triples}
        if not set(result.cited_source_ids) <= sources:
            raise ValueError("answer cites a source outside the extracted evidence")
        if verify:
            reviewed = (verifier or self.model).complete_json(
                "Verify every factual claim in the draft answer against the supplied atomic triples and their evidence. Revise the answer to remove unsupported claims. Return strict JSON with supported (boolean), answer (string), and cited_source_ids (list). Set supported=false if no factual answer remains. Never add a source or fact.",
                json.dumps({
                    "question": question,
                    "draft_answer": result.answer,
                    "triples": [triple.model_dump() for triple in triples],
                }, ensure_ascii=False),
            )
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


__all__ = [
    "AnswerResult",
    "ExtractedTriple",
    "ExtractionEngine",
    "ExtractionResult",
    "ModelClient",
    "ModelFailure",
    "OpenAICompatibleModel",
    "SchemaInference",
]
