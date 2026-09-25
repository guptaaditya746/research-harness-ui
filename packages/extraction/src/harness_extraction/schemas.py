"""Pydantic models for the structured replies of extraction calls."""

from __future__ import annotations

from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


class SchemaInference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str
    gloss: str
    classes: list[str]
    properties: list[str]
    class_definitions: list[dict[str, Any]] = Field(default_factory=list)
    property_definitions: list[dict[str, Any]] = Field(default_factory=list)

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


__all__ = [
    "AnswerResult", "EDCDefinition", "EDCDefinitionResult", "EDCOpenTriple", "ExtractedTriple",
    "ExtractionResult", "SchemaInference",
]
