"""Request bodies accepted by the API."""

from __future__ import annotations

from harness_schema.models import ConceptChoice, EvaluationSnapshot
from pydantic import BaseModel, ConfigDict


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str
    path: str
    budget: str
    evaluation_mode: bool = False
    snapshot: EvaluationSnapshot | None = None
    # Agent-proposed concept chosen by the user; required before agentic research.
    concept: ConceptChoice | None = None
    parent_run_id: str | None = None
    origin: str = "private"
    deadline_seconds: int | None = None


class RatingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rating: str


class CandidateUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ambiguous: bool | None = None
    readings: list[dict] | None = None
    gold_triples: list[dict] | None = None
    gold_claims: list[dict] | None = None
    paper_ids: list[str] | None = None
    notes: str | None = None
    status: str | None = None
    category: str | None = None
    gold_version: str | None = None
    corpus_fingerprint: str | None = None
    ontology_hash: str | None = None
    reviewer: str | None = None
    reviewed_at: str | None = None
    accepted_readings: list[dict] | None = None
    evidence_refs: list[str] | None = None


class ReadingPick(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pick: int


class EvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gold_triples: list[dict]
    gold_claims: list[dict] | None = None


class BenchmarkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repetitions: int = 1
    category: str | None = None


class BenchmarkExecuteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str
    path: str
    budget: str = "slow"
    repetitions: int = 3
    snapshot: EvaluationSnapshot | None = None
    # Agentic only: the concept to research. When omitted, each repetition lets
    # the agent propose concepts and researches its top-ranked grounded one.
    concept: ConceptChoice | None = None


class SettingsUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    baseline: str | None = None
    agentic: str | None = None


class RelationReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str
