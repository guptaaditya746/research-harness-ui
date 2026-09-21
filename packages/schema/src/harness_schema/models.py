"""The handoff run JSON, with v2 terminal metadata added explicitly."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Triple(Contract):
    s: str
    p: str
    o: str
    support: str
    evidence: str
    source: str
    triple_id: str | None = Field(default=None, alias="tripleId")


class Axiom(Contract):
    property: str
    domain: str
    range: str
    sentence: str


class Metrics(Contract):
    disagreement: float | None = None
    precision: tuple[int, int] | None = None
    recall: tuple[int, int] | None = None


class Reading(Contract):
    label: str
    gloss: str
    classes: list[str]
    axioms: list[tuple[str, str, str]]
    sparql: str
    answer: str
    triples: list[Triple]
    metrics: Metrics
    status: Literal["proposed", "answered", "incomplete"] = "proposed"
    op: str = ""
    properties: list[str] = Field(default_factory=list)


class TraceRow(Contract):
    at_s: float = Field(alias="atS")
    action: str
    returned: str
    reading: int | None = None
    span_id: str | None = Field(default=None, alias="spanId")


class Run(Contract):
    id: str
    started_at: datetime = Field(alias="startedAt")
    duration_s: float = Field(alias="durationS")
    question: str
    path: Literal["agentic", "baseline", "baseline-enhanced"]
    budget: Literal["fast", "slow"]
    status: Literal["answered", "awaiting_reading", "incomplete", "stopped"]
    readings: list[Reading] = Field(default_factory=list)
    chosen_reading: int | None = Field(default=None, alias="chosenReading")
    plans: list[str] = Field(default_factory=list)
    trace: list[TraceRow] = Field(default_factory=list)
    rating: Literal["useful", "partial", "not_useful"] | None = None
    reason: str | None = None
    coverage: float | None = None
    theta: float = 0.30
    model_agent: str = Field(default="", alias="modelAgent")
    model_small: str = Field(default="", alias="modelSmall")
    prompt_hashes: dict[str, str] = Field(default_factory=dict, alias="promptHashes")
    artifacts: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_state(self) -> "Run":
        if self.status == "incomplete" and not self.reason:
            raise ValueError("incomplete runs require a reason")
        if self.status == "answered" and (
            self.chosen_reading is None or self.chosen_reading >= len(self.readings)
        ):
            raise ValueError("answered runs require a chosen reading")
        if self.chosen_reading is not None and self.chosen_reading < 0:
            raise ValueError("chosenReading must be non-negative")
        if self.coverage is not None and not 0 <= self.coverage <= 1:
            raise ValueError("coverage must be between zero and one")
        return self


AGENTIC_ARTIFACTS = (
    "plans.json",
    "disagreement.json",
    "readings.json",
    "subschema.ttl",
    "query.rq",
    "coverage_plan.json",
    "ledger.json",
    "graph.trig",
    "triples.ttl",
    "shacl_report.ttl",
    "answer.md",
    "citations.json",
    "verification.json",
    "trace.json",
    "run.json",
)
BASELINE_ARTIFACTS = tuple(
    name
    for name in AGENTIC_ARTIFACTS
    if name not in {"plans.json", "disagreement.json", "readings.json", "verification.json"}
)
ENHANCED_BASELINE_ARTIFACTS = BASELINE_ARTIFACTS + ("retrieval.json", "extraction_batches.json", "seed_manifest.json")


def required_artifacts(path: str) -> tuple[str, ...]:
    match path:
        case "agentic":
            return AGENTIC_ARTIFACTS
        case "baseline":
            return BASELINE_ARTIFACTS
        case "baseline-enhanced":
            return ENHANCED_BASELINE_ARTIFACTS
        case _:
            raise ValueError(f"unknown path: {path}")


def terminal_scene(run: Run) -> str:
    """Exhaustive mapping from v2 terminal states to UI scenes."""
    match run.status:
        case "answered":
            return "agentic_result" if run.path == "agentic" else "baseline_result"
        case "awaiting_reading":
            return "readings"
        case "incomplete":
            return "failed"
        case "stopped":
            return "idle"
        case _:
            raise AssertionError(f"unhandled state {run.status}")
