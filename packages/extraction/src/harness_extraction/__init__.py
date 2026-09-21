"""Shared structured model calls for schema inference, extraction, and answers."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Protocol

import httpx
from harness_semantic import Subschema
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
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

    def answer(
        self, question: str, inferred: SchemaInference, triples: list[ExtractedTriple]
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
