"""The handoff run JSON, with v2 terminal metadata added explicitly."""

from datetime import datetime
from typing import Any, Literal

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


class ClaimField(Contract):
    value: str
    source_ids: list[str] = Field(alias="sourceIds")
    quote: str | None = None
    evidence_ids: list[str] = Field(default_factory=list, alias="evidenceIds")


class EvidenceAnchor(Contract):
    """A reproducible source location, independent of the tool that found it."""
    id: str
    kind: Literal["dblp_fact", "passage", "table_cell", "figure_annotation", "citation_fact"]
    paper_id: str | None = Field(default=None, alias="paperId")
    source_id: str = Field(alias="sourceId")
    locator: dict[str, str | int] = Field(default_factory=dict)
    text: str | None = None


class ClaimSubgraph(Contract):
    id: str
    kind: Literal["MetadataClaim", "ResultClaim", "ComparisonClaim", "MethodClaim", "LimitationClaim", "RankingClaim", "SynthesisClaim"]
    paper_id: str = Field(alias="paperId")
    fields: dict[str, ClaimField]
    source_ids: list[str] = Field(alias="sourceIds")
    evidence: dict[str, str]
    evidence_anchors: list[EvidenceAnchor] = Field(default_factory=list, alias="evidenceAnchors")
    supported_by: list[str] = Field(default_factory=list, alias="supportedBy")
    status: Literal["accepted", "partial"] = "accepted"

    @model_validator(mode="after")
    def validate_provenance(self) -> "ClaimSubgraph":
        if not self.fields or not self.source_ids:
            raise ValueError("claim requires fields and provenance")
        if any(not field.value.strip() or field.value.strip().casefold() in {"null", "none", "n/a"}
               for field in self.fields.values()):
            raise ValueError("claim contains a placeholder field")
        if any(not field.source_ids or not set(field.source_ids) <= set(self.source_ids)
               for field in self.fields.values()):
            raise ValueError("field provenance must belong to the claim")
        if not set(self.source_ids) <= set(self.evidence):
            raise ValueError("claim evidence missing a cited source")
        anchor_ids = {anchor.id for anchor in self.evidence_anchors}
        if any(field.evidence_ids and not set(field.evidence_ids) <= anchor_ids for field in self.fields.values()):
            raise ValueError("field evidence must belong to the claim")
        if self.kind == "SynthesisClaim" and not self.supported_by:
            raise ValueError("synthesis claims require supporting claims")
        return self


class ClaimRelation(Contract):
    source_claim: str = Field(alias="sourceClaim")
    target_claim: str = Field(alias="targetClaim")
    relation: Literal["supports", "contradicts", "qualifies", "sameContext", "differentContext", "possibleConflict", "confirmedConflict", "equivalentTo"]
    reason: str


class Axiom(Contract):
    property: str
    domain: str
    range: str
    sentence: str


class Metrics(Contract):
    disagreement: float | None = None
    precision: tuple[int, int] | None = None
    recall: tuple[int, int] | None = None


class GraphNode(Contract):
    """One entity in the answer graph; the same entity appears exactly once."""

    id: str
    label: str
    kind: Literal["paper", "author", "value", "entity", "claim", "passage", "computed"]
    iri: str | None = None


Origin = Literal["dblp", "core", "invented"]


class GraphEdge(Contract):
    """A source fact or claim field (``fact``/``claim``/``evidence``) or a computed step (``derived``).

    ``origin`` says where a fact edge's predicate comes from: the DBLP ontology, the harness core
    vocabulary, or invented for this question (None for claim, evidence and derived edges).
    """

    id: str
    source: str
    target: str
    label: str
    kind: Literal["fact", "claim", "evidence", "derived"]
    source_ids: list[str] = Field(default_factory=list, alias="sourceIds")
    evidence: str | None = None
    origin: Origin | None = None


class SchemaTerm(Contract):
    """One class or property of a reading's subschema, as declared in OWL 2, with its origin."""

    name: str                                   # prefixed: dblp:authoredBy, hrn:addresses
    iri: str
    kind: Literal["class", "object", "datatype"]
    origin: Origin
    parent: str | None = None                   # prefixed name of the superclass / superproperty
    domain: str | None = None
    range: str | None = None
    definition: str | None = None


class ComputedResult(Contract):
    """A conclusion the harness computed from source facts; never presented as a source fact."""

    id: str
    label: str
    operation: str
    about: str
    derived_from: list[str] = Field(default_factory=list, alias="derivedFrom")


class AnswerGraph(Contract):
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    computed: list[ComputedResult] = Field(default_factory=list)


class Check(Contract):
    """One stage's outcome, so a failure names the stage that failed."""

    name: str
    status: Literal["pass", "warn", "fail", "skipped", "info"]
    detail: str = ""


class AnswerSentence(Contract):
    """One sentence of a verified answer with the sources it cites and the verifier's verdict (Phase 5.1).

    ``accepted``: a supplied triple states it. ``removed``: no triple supports it; it is not in the answer.
    ``from_facts``: the composed answer had no supported sentence after one repair, so the answer states
    the extracted facts themselves.
    """

    text: str
    source_ids: list[str] = Field(default_factory=list, alias="sourceIds")
    verdict: Literal["accepted", "removed", "from_facts"]
    reason: str = ""


class Reading(Contract):
    label: str
    gloss: str
    classes: list[str]
    axioms: list[tuple[str, str, str]]
    sparql: str
    answer: str
    triples: list[Triple]
    claims: list[ClaimSubgraph] = Field(default_factory=list)
    claim_relations: list[ClaimRelation] = Field(default_factory=list, alias="claimRelations")
    answer_claim_ids: list[str] = Field(default_factory=list, alias="answerClaimIds")
    metrics: Metrics
    status: Literal["proposed", "answered", "incomplete"] = "proposed"
    op: str = ""
    properties: list[str] = Field(default_factory=list)
    # What the answer actually computed or asked for (e.g. "distinct papers per author").
    measure: str | None = None
    graph: AnswerGraph | None = None
    # The subschema's terms with their origin (DBLP / core / invented), for the Subschema panel.
    schema_terms: list[SchemaTerm] = Field(default_factory=list, alias="schemaTerms")
    # Verified answers only: every sentence the composer wrote, kept or removed, with its sources.
    answer_sentences: list[AnswerSentence] = Field(default_factory=list, alias="answerSentences")


class TraceRow(Contract):
    at_s: float = Field(alias="atS")
    action: str
    returned: str
    reading: int | None = None
    span_id: str | None = Field(default=None, alias="spanId")
    purpose: str | None = None
    target_claim: str | None = Field(default=None, alias="targetClaim")
    expected_evidence_type: str | None = Field(default=None, alias="expectedEvidenceType")


class EvaluationSnapshot(Contract):
    corpus_snapshot_id: str = Field(alias="corpusSnapshotId")
    ontology_version: str = Field(alias="ontologyVersion")
    claim_schema_version: str = Field(alias="claimSchemaVersion")
    gold_version: str | None = Field(default=None, alias="goldVersion")
    retrieval_index_version: str = Field(alias="retrievalIndexVersion")


class ConceptChoice(Contract):
    """The research interpretation a run is bound to (agentic path)."""

    label: str
    gloss: str
    classes: list[str] = Field(default_factory=list)
    search_terms: list[str] = Field(default_factory=list, alias="searchTerms")
    evidence_hits: int | None = Field(default=None, alias="evidenceHits")
    # Passages that support this reading (grounding); research starts from them.
    sample_sources: list[str] = Field(default_factory=list, alias="sampleSources")
    auto_selected: bool = Field(default=False, alias="autoSelected")


class Run(Contract):
    id: str
    started_at: datetime = Field(alias="startedAt")
    duration_s: float = Field(alias="durationS")
    question: str
    path: Literal["agentic", "baseline", "baseline-enhanced", "external"]
    budget: Literal["fast", "slow"]
    status: Literal["answered", "corpus_absent", "no_evidence_found", "awaiting_reading", "incomplete", "stopped"]
    readings: list[Reading] = Field(default_factory=list)
    chosen_reading: int | None = Field(default=None, alias="chosenReading")
    plans: list[str] = Field(default_factory=list)
    trace: list[TraceRow] = Field(default_factory=list)
    rating: Literal["useful", "partial", "not_useful"] | None = None
    reason: str | None = None
    producing_path: Literal["agentic", "baseline", "baseline-enhanced", "baseline_fallback", "external"] | None = Field(
        default=None, alias="producingPath")
    execution_status: Literal["completed", "degraded", "failed"] | None = Field(default=None, alias="executionStatus")
    evidence_surface: str | None = Field(default=None, alias="evidenceSurface")
    search_coverage: dict[str, str | int | bool] | None = Field(default=None, alias="searchCoverage")
    evaluation_mode: bool = Field(default=False, alias="evaluationMode")
    snapshot: EvaluationSnapshot | None = None
    # What the run read: "corpus" (it searched the papers) or "evidence_doc" (the question's evidence
    # document D was its input; ``evidence_doc`` records which file). Scores are compared within one input.
    run_input: Literal["corpus", "evidence_doc"] = Field(default="corpus", alias="runInput")
    evidence_doc: dict[str, Any] | None = Field(default=None, alias="evidenceDoc")
    # The settings that are editable in the UI, as they were when the run started.
    settings: dict[str, str] = Field(default_factory=dict)
    concept: ConceptChoice | None = None
    concepts: list[ConceptChoice] = Field(default_factory=list)
    parent_run_id: str | None = Field(default=None, alias="parentRunId")
    checks: list[Check] = Field(default_factory=list)
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
        if self.status in {"corpus_absent", "no_evidence_found"} and (self.evidence_surface is None or self.search_coverage is None):
            raise ValueError("corpus absence requires an evidence surface and coverage")
        if self.chosen_reading is not None and self.chosen_reading < 0:
            raise ValueError("chosenReading must be non-negative")
        if self.coverage is not None and not 0 <= self.coverage <= 1:
            raise ValueError("coverage must be between zero and one")
        return self


AGENTIC_ARTIFACTS = (
    "plans.json",
    "disagreement.json",
    "readings.json",
    "research_ledger.json",
    "context_checkpoints.json",
    "subschema.ttl",
    "query.rq",
    "coverage_plan.json",
    "ledger.json",
    "graph.trig",
    "triples.ttl",
    "claims.json",
    "shacl_report.ttl",
    "answer.md",
    "citations.json",
    "verification.json",
    "trace.json",
    "tool_calls.jsonl",
    "agent_draft.md",
    # The agent's pipeline is the enhanced one: keep its retrieval and extraction reports
    # (graph-shape reads the extraction funnel from extraction_batches.json).
    "retrieval.json",
    "extraction_batches.json",
    "seed_manifest.json",
    # Every extracted triple with the evidence review's verdict, not only those the answer used.
    "evidence_graph.json",
    "run.json",
)
ENHANCED_ONLY = ("retrieval.json", "extraction_batches.json", "seed_manifest.json", "evidence_graph.json")
BASELINE_ARTIFACTS = tuple(
    name
    for name in AGENTIC_ARTIFACTS
    if name not in {"plans.json", "disagreement.json", "readings.json", "research_ledger.json", "context_checkpoints.json",
                    "verification.json", "agent_draft.md", *ENHANCED_ONLY}
)
ENHANCED_BASELINE_ARTIFACTS = BASELINE_ARTIFACTS + ENHANCED_ONLY
# The external harness (a separate repo run as a subprocess): its own result and trajectory are kept too.
EXTERNAL_ARTIFACTS = BASELINE_ARTIFACTS + ("external_result.json", "external_trajectory.json")


def required_artifacts(path: str) -> tuple[str, ...]:
    match path:
        case "agentic":
            return AGENTIC_ARTIFACTS
        case "baseline":
            return BASELINE_ARTIFACTS
        case "baseline-enhanced":
            return ENHANCED_BASELINE_ARTIFACTS
        case "external":
            return EXTERNAL_ARTIFACTS
        case _:
            raise ValueError(f"unknown path: {path}")


def terminal_scene(run: Run) -> str:
    """Exhaustive mapping from v2 terminal states to UI scenes."""
    match run.status:
        case "answered":
            return "agentic_result" if run.path == "agentic" else "baseline_result"
        case "corpus_absent" | "no_evidence_found":
            return "agentic_result" if run.path == "agentic" else "baseline_result"
        case "awaiting_reading":
            return "readings"
        case "incomplete":
            return "failed"
        case "stopped":
            return "idle"
        case _:
            raise AssertionError(f"unhandled state {run.status}")
