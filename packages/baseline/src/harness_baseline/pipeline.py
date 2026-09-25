"""Baseline-20: the fixed four-node LangGraph workflow shared by the Fast and Adaptive paths.

Nodes: retrieve -> infer_schema -> extract_build -> query_answer. Fact questions are answered
from verified facts before the graph runs (see ``facts``). The steps live in mixins:
``retrieval``, ``research_steps``, ``facts`` and ``outcomes``.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from harness_corpus.store import CorpusStore
from harness_extraction import (
    AnswerResult,
    ExtractedTriple,
    ExtractionEngine,
    ModelFailure,
    SchemaInference,
)
from harness_observability import record_stage
from harness_schema.models import (
    Check,
    ClaimSubgraph,
    TraceRow,
)
from langgraph.graph import END, START, StateGraph

from .facts import FactAnswers
from .outcomes import RunOutcomes
from .planner import QueryPlanner
from .research_steps import ResearchSteps
from .retrieval import RetrievalStep
from .state import (
    BaselineResult,
    BaselineState,
    CorpusAbsent,
    NoEvidenceFound,
    NoResults,
    TraceSink,
)


class Baseline20(RetrievalStep, ResearchSteps, FactAnswers, RunOutcomes):
    """One research run: plan the question, answer facts directly, else run the graph."""

    def __init__(
        self,
        corpus: CorpusStore,
        extraction: ExtractionEngine,
        retrieve_k: int = 20,
        enhanced: bool = False,
        edc: bool = False,
        trace_sink: TraceSink | None = None,
        refine: bool = False,
        refinement_model: Any | None = None,
        claim_lookup: Callable[[str], list[ClaimSubgraph]] | None = None,
        relation_proposer: Callable[[str, str, str | None, str, str], dict] | None = None,
        seed_passage_ids: list[str] | None = None,
        planner: QueryPlanner | None = None,
        max_passages: int | None = None,
    ) -> None:
        if max_passages is not None and max_passages < 1:
            raise ValueError("max_passages must be positive")
        self.corpus = corpus
        self.planner = planner
        self.extraction = extraction
        self.retrieve_k = retrieve_k
        self.enhanced = enhanced
        self.edc = edc
        self.trace_sink = trace_sink or (lambda _: None)
        self.refine = refine
        self.refinement_model = refinement_model
        self.claim_lookup = claim_lookup
        self.relation_proposer = relation_proposer
        self.seed_passage_ids = seed_passage_ids or []
        # Findings the agentic path hands over (statements + evidence ids); see _research_notes.
        self.agent_findings: list[dict[str, Any]] = []
        self.max_passages = max_passages
        self.started = 0.0
        self.trace: list[TraceRow] = []
        self.current_run_id: str | None = None
        self.concept: dict[str, Any] | None = None
        self.checks: list[Check] = []
        self.stage = "question understanding"
        # How the last run was answered: "verified_facts" or "research".
        self.last_route: str | None = None
        self.graph = self._graph()

    @staticmethod
    def _research_question(state: BaselineState) -> str:
        """The question as given to models, with the chosen interpretation attached."""
        text = state["question"]
        concept = state.get("concept")
        if concept:
            text += (
                "\n\nResearch interpretation chosen by the user: "
                f"{concept.get('label', '')}: {concept.get('gloss', '')}"
            )
        return text

    @staticmethod
    def _research_notes(state: BaselineState) -> list[str]:
        """The agent's findings as unverified notes for schema inference and answer composition.

        Statements only: the ids are left out so the answer cannot cite a passage no triple uses.
        Extraction and review never see them, so every triple still comes from a passage.
        """
        return [str(item["statement"]) for item in state.get("agent_findings") or [] if item.get("statement")]

    def _check(self, name: str, status: str, detail: str = "") -> None:
        """Record one stage's outcome (pass/warn/fail/skipped) for the run's checks."""
        self.checks = [check for check in self.checks if check.name != name]
        self.checks.append(Check(name=name, status=status, detail=detail))

    def _emit(self, action: str, returned: str) -> None:
        row = TraceRow(
            atS=round(time.monotonic() - self.started, 3), action=action, returned=returned
        )
        self.trace.append(row)
        self.trace_sink(row)
        record_stage(action, returned, run_id=self.current_run_id)

    def _graph(self):
        graph = StateGraph(BaselineState)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("infer_schema", self._infer_schema)
        graph.add_node("extract_build", self._extract_build)
        graph.add_node("query_answer", self._query_answer)
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "infer_schema")
        graph.add_edge("infer_schema", "extract_build")
        graph.add_edge("extract_build", "query_answer")
        graph.add_edge("query_answer", END)
        return graph.compile()

    def run(
        self,
        question: str,
        budget: str = "fast",
        run_id: str | None = None,
        concept: dict[str, Any] | None = None,
    ) -> BaselineResult:
        """Answer ``question``; an optional ``concept`` scopes retrieval and extraction.

        The question text is never rewritten. A concept (label, gloss, classes,
        search_terms) chosen by the user on the agentic path is structured input:
        its search terms widen retrieval, its description ranks evidence, and it
        is given to the extraction model as the research interpretation.
        """
        run_id = run_id or f"r-{uuid.uuid4().hex[:8]}"
        started_at = datetime.now(UTC)
        self.started = time.monotonic()
        self.current_run_id = run_id
        self.trace = []
        self.concept = concept or None
        self.checks = []
        self.stage = "question understanding"
        try:
            plan = self._plan(question)
            self._emit(
                "Understanding question",
                f"{plan.operation}" + (f" · {plan.rationale}" if plan.rationale else ""),
            )
            self._check("question understanding", "warn" if plan.source == "fallback" else "pass",
                        plan.rationale if plan.source == "fallback" else
                        f"{plan.operation}" + (f" · measure: {plan.stated_measure()}" if plan.stated_measure() else ""))
            papers, unresolved = self._resolve_papers(plan)
            if unresolved:
                self._emit("Paper not found in corpus titles", "; ".join(unresolved))
            direct: tuple[AnswerResult, list[ExtractedTriple]] | None = None
            lookup_started = time.monotonic()
            self._computed = []
            if plan.operation != "research" and not unresolved:
                self.stage = "verified facts"
                direct = self._fact_query(plan, papers)
                if direct is None:
                    self._emit("Verified facts do not cover this", "continuing with full-text research")
                    self._check("verified facts", "skipped", "the facts needed are not indexed; used full text")
            elif unresolved:
                self._check("verified facts", "skipped", f"paper not found: {'; '.join(unresolved)}")
            self.last_route = "verified_facts" if direct is not None else "research"
            if direct is not None:
                answer, triples = direct
                properties = list(dict.fromkeys(item.predicate for item in triples))
                inferred = SchemaInference(
                    label="Corpus facts", gloss="Verified facts relevant to the question.",
                    classes=["Paper", "Author"] if "authoredBy" in properties else ["Paper"],
                    properties=properties,
                )
                self._emit("Retrieving verified corpus facts", f"{len(triples)} facts")
                self._emit("Selecting atomic triples", f"{len(triples)} question-relevant triples")
                self._check("verified facts", "pass", f"{len(triples)} DBLP facts / resolved references")
                self._check("retrieval", "skipped", "answered from verified facts")
                self._check("extraction", "skipped", "answered from verified facts")
                state: BaselineState = {
                    "question": question, "passages": [], "inferred": inferred,
                    "extracted": triples, "answer": answer, "plan": plan, "paper_ids": papers,
                    "computed": list(self._computed),
                    "retrieval": {
                        "queries": [], "stop_reason": "verified_facts",
                        "plan": plan.model_dump(),
                        "fact_lookup_ms": round((time.monotonic() - lookup_started) * 1000, 1),
                    },
                    "extraction_batches": [{"stage": "verified_facts", "triples": len(triples)}],
                }
            else:
                state = self.graph.invoke({
                    "question": question, "concept": self.concept, "plan": plan,
                    "paper_ids": papers, "computed": [], "agent_findings": list(self.agent_findings),
                })
            self.stage = "triple provenance"
            run, artifacts = self._success(run_id, started_at, question, budget, state)
        except CorpusAbsent as exc:
            self._check(self.stage, "pass", f"completed: nothing matches on {exc.evidence_surface}")
            run, artifacts = self._absence(run_id, started_at, question, budget, exc)
        except NoEvidenceFound as exc:
            self._check("retrieval", "fail", f"full-text search found no matching passage: {exc}")
            run, artifacts = self._no_evidence(run_id, started_at, question, budget, exc)
        except NoResults as exc:
            self._check(self.stage, "fail", str(exc))
            run, artifacts = self._failure(
                run_id, started_at, question, budget, "no_results", str(exc)
            )
        except ModelFailure as exc:
            self._check(self.stage, "fail", f"model call failed: {exc}")
            run, artifacts = self._failure(
                run_id, started_at, question, budget, "provider_error", str(exc)
            )
        except ValueError as exc:
            reason = (
                "query_rejected"
                if "SPARQL" in str(exc) or "query" in str(exc)
                else "schema_rejected"
            )
            self._check(self.stage, "fail", f"{reason}: {exc}")
            run, artifacts = self._failure(run_id, started_at, question, budget, reason, str(exc))
        run = run.model_copy(update={"checks": list(self.checks)})
        return {"run": run, "artifacts": artifacts}

