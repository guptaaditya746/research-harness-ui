"""Agentic research: an MCP tool-using agent seeds the shared evidence pipeline.

One run, after the user has picked a concept reading:

1. ``_check_heavy_load``: defer when the agent itself runs on a saturated Heavy.
2. ``_mcp_research``: the agent (Lite) researches with the corpus MCP tools, records
   findings (``record_finding``) and returns a JSON decision naming the evidence it found.
3. ``_collect_findings``: evidence the session observed and the agent's validated findings.
4. ``_read_papers``: parallel paper readers (``readers``) add findings for the papers with
   the most evidence (``HARNESS_SUBAGENT_READERS``).
5. ``_select_seeds``: one passage per finding first, then other observed evidence, capped
   (``HARNESS_AGENTIC_MAX_MCP_SEEDS``); recover passages when the decision failed.
6. ``_run_shared_pipeline``: the baseline pipeline (Heavy) runs with those seeds and the
   findings as unverified notes.
7. ``_review_decision``: validate the decision, run the independent verifier and,
   when the agent only confirmed preliminary claims, narrow the answer to them.
8. ``_assemble``: checks, run record and artifacts.

Everything the steps share lives on one ``_Session`` object.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from time import monotonic
from typing import Any

from harness_baseline import Baseline20
from harness_config import settings
from harness_observability import record_tool_call
from harness_schema.models import Check, ConceptChoice, Run, TraceRow, required_artifacts

from .evidence import evidence_ids
from .load import heavy_queue_depth
from .mcp import MCPToolClient, MCPUnavailable, id_list, validate_findings
from .proposal import ConceptProposalStep
from .readers import choose_papers, run_readers
from .verification import independent_verification

# Bookkeeping records in tool_calls that are not corpus research.
NON_CORPUS_RECORDS = {"load_skill", "context_checkpoint", "mcp_session", "record_finding", "paper_reader"}

_TOOL_ACTIONS = {
    "search_evidence": "Searching corpus evidence",
    "read_evidence": "Inspecting supporting evidence",
    "structured_facts": "Checking canonical metadata",
    "citations": "Checking citation evidence",
    "tables": "Inspecting table evidence",
    "evidence_for_claim": "Verifying claim context",
    "load_skill": "Loading research skill",
    "record_finding": "Recording a finding",
    "paper_reader": "Reading a paper in parallel",
    "context_checkpoint": "Context checkpoint",
}


@dataclass
class _Session:
    """State shared by the steps of one agentic run."""

    question: str
    budget: str
    run_id: str
    concept: ConceptChoice
    concept_input: dict
    started: float
    tool_calls: list[dict] = field(default_factory=list)
    trace: list[TraceRow] = field(default_factory=list)
    mcp_error: str | None = None
    agent_final: str = ""
    agent_decision: dict = field(default_factory=dict)
    shown_ids: set[str] = field(default_factory=set)
    shown_claim_ids: set[str] = field(default_factory=set)
    agent_checks: list[Check] = field(default_factory=list)
    independent: dict | None = None
    heavy_busy: bool = False
    corpus_calls: list[dict] = field(default_factory=list)
    observed: set[str] = field(default_factory=set)
    selected_ids: list[str] = field(default_factory=list)
    notes: list[dict] = field(default_factory=list)
    findings: list[dict] = field(default_factory=list)
    evidence_limit: int = 6
    used_for_verification: list[str] = field(default_factory=list)
    baseline_started_s: float = 0.0
    base_run: Run | None = None
    result: dict | None = None

    def elapsed(self) -> float:
        return round(monotonic() - self.started, 3)


class AgenticResearch(ConceptProposalStep):
    """Agent-path runner sharing evidence and graph invariants with baseline."""

    def __init__(
        self,
        baseline: Baseline20,
        trace_sink: Callable[[TraceRow], None] | None = None,
        agent_model: object | None = None,
        high_model: object | None = None,
    ):
        self.baseline = baseline
        self.trace_sink = trace_sink or (lambda _: None)
        self.agent_model = agent_model or baseline.extraction.model
        self.high_model = high_model or self.agent_model

    def run(
        self, question: str, budget: str, run_id: str,
        concept: ConceptChoice | dict | None = None,
    ):
        """Run agentic research.

        Without a chosen ``concept`` the agent first proposes 3-4 concept
        readings and the run stops at ``awaiting_reading``. Research proceeds
        only once a concept has been chosen; the question itself is never
        rewritten, the concept travels alongside it as structured input.
        """
        if concept is None:
            return self.propose(question, budget, run_id)
        concept = (
            concept if isinstance(concept, ConceptChoice) else ConceptChoice.model_validate(concept)
        )
        session = _Session(
            question=question, budget=budget, run_id=run_id, concept=concept,
            concept_input=concept.model_dump(by_alias=False), started=monotonic(),
        )
        self._check_heavy_load(session)
        # Search through MCP before the shared pipeline verifies its answer.
        # Otherwise a rejected draft prevents the agent from gathering the
        # very evidence that could make the answer supportable.
        if not session.heavy_busy:
            self._mcp_research(session)
        if not session.heavy_busy and not session.agent_decision and not session.mcp_error:
            session.mcp_error = "agent_returned_no_research_decision"
        self._collect_findings(session)
        if not session.heavy_busy:
            self._read_papers(session)
        self._select_seeds(session)
        self._run_shared_pipeline(session)
        self._review_decision(session)
        if session.mcp_error:
            self._record_session_error(session)
        return self._assemble(session)

    # ─── Shared helpers ──────────────────────────────────────────────────────

    def _trace(self, session: _Session, row: TraceRow) -> None:
        session.trace.append(row)
        self.trace_sink(row)

    def _tool_sink(self, session: _Session, record: dict) -> None:
        session.tool_calls.append(record)
        record_tool_call(record, run_id=session.run_id)
        self._trace(session, TraceRow(
            atS=session.elapsed(),
            action=_TOOL_ACTIONS.get(record["tool"], "Researching corpus evidence"),
            returned=(
                f"{record.get('status')} · {record.get('result_count', 0)} items · "
                f"{record.get('duration_ms', 0)} ms"
            ),
            purpose=record.get("purpose"),
            expectedEvidenceType=record.get("expected_evidence_type"),
        ))

    @staticmethod
    def _heavy_queue_depth() -> int | None:
        """Requests waiting on the high model, or None when load is unknown (see ``load``)."""
        return heavy_queue_depth()

    @staticmethod
    def _evidence_ids(value: object) -> set[str]:
        return evidence_ids(value)

    def _independent_verification(
        self, question: str, decision: dict, verified: list, evidence: list,
        corpus_calls: list[dict], base_run: Run,
    ) -> tuple[dict | None, Check]:
        verifier = getattr(self.baseline, "refinement_model", None) or self.agent_model
        return independent_verification(verifier, question, decision, verified, evidence,
                                        corpus_calls, base_run)

    # ─── Steps ───────────────────────────────────────────────────────────────

    def _check_heavy_load(self, session: _Session) -> None:
        config = settings()
        queue_depth = self._heavy_queue_depth()
        # Granite controls MCP research. Qwen queue pressure must not turn an
        # otherwise healthy Granite-led agentic execution into a fallback.
        controller_uses_heavy = getattr(self.agent_model, "model", "") == config.model_high
        session.heavy_busy = (
            controller_uses_heavy
            and queue_depth is not None
            and queue_depth >= config.heavy_queue_threshold
        )
        if session.heavy_busy:
            self.baseline.refinement_model = None
            session.mcp_error = f"heavy_model_busy: {queue_depth} requests waiting"
            self._trace(session, TraceRow(atS=0.0, action="Heavy model deferred",
                                          returned=session.mcp_error))

    def _mcp_research(self, session: _Session) -> None:
        mcp = MCPToolClient(lambda record: self._tool_sink(session, record), agent="supervisor")
        # The same list the tool loop appends to: findings survive a timeout or a failed decision.
        session.notes = mcp.notes
        try:
            grounded_context = self._grounded_context(None, session.concept_input)
            session.shown_ids = self._evidence_ids(grounded_context)
            session.shown_claim_ids = {
                str(claim["id"]) for claim in grounded_context.get("preliminary_claims", [])
            }
            session.agent_final = mcp.run_agent(self.agent_model, session.question, grounded_context)
            if session.agent_final:
                try:
                    start, end = session.agent_final.find("{"), session.agent_final.rfind("}")
                    parsed = json.loads(session.agent_final[start:end + 1])
                    if not isinstance(parsed, dict):
                        raise TypeError("agent decision must be an object")
                    session.agent_decision = parsed
                except (ValueError, TypeError):
                    session.mcp_error = "agent_final_not_structured"
                self._trace(session, TraceRow(atS=session.elapsed(), action="Agent research decision",
                                              returned=session.agent_final[:500]))
        except MCPUnavailable as exc:
            session.mcp_error = str(exc)
            self._trace(session, TraceRow(atS=session.elapsed(), action="MCP unavailable",
                                          returned=str(exc)))

    def _collect_findings(self, session: _Session) -> None:
        session.corpus_calls = [item for item in session.tool_calls
                                if item["tool"] not in NON_CORPUS_RECORDS]
        # Only IDs the agent actually saw in this session (tool results and the
        # preliminary result shown to it) count as evidence; existing somewhere
        # in the corpus is not enough.
        session.observed = session.shown_ids.union(
            *(self._evidence_ids(item.get("result")) for item in session.corpus_calls))
        if settings().agent_handoff == "findings":
            raw = [*session.notes, *(session.agent_decision.get("findings") or []
                                     if isinstance(session.agent_decision.get("findings"), list) else [])]
            findings, _ = validate_findings(raw, session.observed)
            session.findings = self._unique_findings(findings)

    @staticmethod
    def _unique_findings(findings: list[dict]) -> list[dict]:
        unique: dict[str, dict] = {}
        for finding in findings:
            unique.setdefault(" ".join(finding["statement"].casefold().split()), finding)
        return list(unique.values())

    def _read_papers(self, session: _Session) -> None:
        """Parallel paper readers on the agent model (see ``readers``)."""
        config = settings()
        corpus = getattr(self.baseline, "corpus", None)
        if (config.subagent_readers <= 0 or config.agent_handoff != "findings" or corpus is None
                or not callable(getattr(self.agent_model, "complete_json", None))):
            return
        weighted = [(value, 3) for finding in session.findings for value in finding["evidence_ids"]]
        supplied = session.agent_decision.get("additional_evidence_ids")
        weighted += [(value, 2) for value in (id_list(supplied) if isinstance(supplied, list) else [])
                     if value in session.observed]
        weighted += [(value, 1) for value in session.observed]
        papers = choose_papers(corpus, session.question, weighted, config.subagent_readers)
        if not papers:
            return
        started = session.elapsed()
        results = run_readers(self.agent_model, corpus, session.question, session.concept_input, papers,
                              max(1, config.subagent_passages))
        added: list[dict] = []
        for result in results:
            # The passages a reader was shown count as observed in this session.
            session.observed.update(result.passage_ids)
            added.extend({**finding, "source": f"reader:{result.paper_id}"} for finding in result.findings)
            self._tool_sink(session, {
                "call_id": f"reader-{result.paper_id}", "agent": f"reader:{result.paper_id}",
                "tool": "paper_reader", "arguments": {"paper_id": result.paper_id},
                "result": {"passage_ids": result.passage_ids, "findings": result.findings},
                "result_count": len(result.findings), "duration_ms": result.duration_ms,
                "status": "error" if result.error else "ok", **({"error": result.error} if result.error else {}),
                "purpose": "Read one paper's top passages in parallel and report findings.",
                "expected_evidence_type": "research_note",
            })
        session.findings = self._unique_findings([*session.findings, *added])
        session.agent_checks.append(Check(
            name="paper readers", status="pass" if added else "info",
            detail=f"{len(papers)} papers read in parallel in {session.elapsed() - started:.1f}s; "
                   f"{len(added)} findings",
        ))

    def _select_seeds(self, session: _Session) -> None:
        if session.mcp_error and session.corpus_calls:
            self._recover_passages(session)
        supplied_ids = session.agent_decision.get("additional_evidence_ids")
        session.evidence_limit = max(1, settings().agentic_max_mcp_seeds)
        decision_ids = (
            [item for item in id_list(supplied_ids) if item in session.observed]
            if isinstance(supplied_ids, list) else []
        )
        corpus_passages = getattr(getattr(self.baseline, "corpus", None), "by_passage", None)
        # Order: one passage per supervisor finding (every finding reaches the pipeline), the
        # supervisor's decision ids, one passage per paper from the readers, then the rest. Readers
        # can add up to 12 findings; they must not push the supervisor's evidence out of the seeds.
        own = [finding for finding in session.findings if not str(finding.get("source", "")).startswith("reader:")]
        read = [finding for finding in session.findings if str(finding.get("source", "")).startswith("reader:")]
        finding_first = [finding["evidence_ids"][0] for finding in own]
        reader_first: list[str] = []
        reader_papers: set[str] = set()
        for finding in read:
            first = finding["evidence_ids"][0]
            paper = str((corpus_passages or {}).get(first, {}).get("paper_id") or finding.get("source"))
            if paper not in reader_papers:
                reader_papers.add(paper)
                reader_first.append(first)
        rest = [value for finding in own for value in finding["evidence_ids"][1:]]
        rest += [value for finding in read for value in finding["evidence_ids"]]
        selected_ids = list(dict.fromkeys([*finding_first, *decision_ids, *reader_first, *rest]))
        if corpus_passages is not None:
            selected_ids = [item for item in selected_ids if item in corpus_passages]
            priority = [item for item in dict.fromkeys(finding_first) if item in corpus_passages]
            diverse = []
            seen_papers = {str(corpus_passages[source_id]["paper_id"]) for source_id in priority}
            for source_id in selected_ids:
                paper_id = str(corpus_passages[source_id]["paper_id"])
                if paper_id not in seen_papers:
                    diverse.append(source_id)
                    seen_papers.add(paper_id)
            selected_ids = list(dict.fromkeys([*priority, *diverse, *selected_ids]))
        deferred_ids = max(0, len(selected_ids) - session.evidence_limit)
        session.selected_ids = selected_ids[:session.evidence_limit]
        if deferred_ids:
            session.agent_checks.append(Check(
                name="agent evidence budget", status="info",
                detail=f"{len(session.selected_ids)} MCP passages selected; {deferred_ids} deferred for response time",
            ))
        session.agent_checks.append(Check(
            name="agent passage seeds",
            status="pass" if session.selected_ids else "skipped",
            detail=f"{len(session.selected_ids)} MCP passage IDs supplied before retrieval"
            + (f"; {len(session.findings)} findings handed over" if session.findings else ""),
        ))

    def _recover_passages(self, session: _Session) -> None:
        """A failed controller decision stays visible as degraded.

        Recover a small set of actual passage handles from successful tool calls
        so a timeout or malformed final answer does not waste research already
        gathered before schema inference.
        """
        # Evidence the agent already wrote findings about comes first.
        recovered: list[str] = [value for finding in session.findings for value in finding["evidence_ids"]][:12]

        def passage_ids(value: object) -> list[str]:
            if isinstance(value, dict):
                found = [str(value["passage_id"])] if value.get("passage_id") else []
                return found + [handle for child in value.values() for handle in passage_ids(child)]
            if isinstance(value, list):
                return [handle for child in value for handle in passage_ids(child)]
            return []

        for call in session.corpus_calls:
            if call.get("status") != "ok" or call.get("tool") not in {
                "search_evidence", "read_evidence", "rank_evidence", "passages",
            }:
                continue
            recovered.extend(passage_ids(call.get("result"))[:2])
            recovered = list(dict.fromkeys(recovered))[:12]
            if len(recovered) == 12:
                break
        if recovered:
            session.agent_decision = {
                "verified_claim_ids": [],
                "additional_evidence_ids": recovered,
                "reasoning_summary": "Recovered observed MCP passages after the controller failed.",
            }
            session.agent_checks.append(Check(
                name="agent passage recovery", status="warn",
                detail=f"{len(recovered)} observed passage IDs recovered from MCP tool results",
            ))

    def _run_shared_pipeline(self, session: _Session) -> None:
        previous_seed_ids = list(getattr(self.baseline, "seed_passage_ids", []))
        previous_findings = list(getattr(self.baseline, "agent_findings", []))
        if hasattr(self.baseline, "seed_passage_ids"):
            self.baseline.seed_passage_ids = list(dict.fromkeys([*previous_seed_ids, *session.selected_ids]))
        if hasattr(self.baseline, "agent_findings"):
            self.baseline.agent_findings = list(session.findings)
        session.baseline_started_s = monotonic() - session.started
        original_trace_sink = getattr(self.baseline, "trace_sink", None)
        if callable(original_trace_sink):
            offset = session.baseline_started_s
            self.baseline.trace_sink = lambda row: original_trace_sink(
                row.model_copy(update={"at_s": round(row.at_s + offset, 3)})
            )
        try:
            session.result = self.baseline.run(session.question, session.budget, session.run_id,
                                               concept=session.concept_input)
        finally:
            if hasattr(self.baseline, "seed_passage_ids"):
                self.baseline.seed_passage_ids = previous_seed_ids
            if hasattr(self.baseline, "agent_findings"):
                self.baseline.agent_findings = previous_findings
            if callable(original_trace_sink):
                self.baseline.trace_sink = original_trace_sink
        session.base_run = session.result["run"]

    def _review_decision(self, session: _Session) -> None:
        base_run = session.base_run
        assert base_run is not None and session.result is not None
        decision = session.agent_decision
        if not decision or session.mcp_error:
            return
        verified = decision.get("verified_claim_ids")
        supplied = decision.get("additional_evidence_ids")
        if isinstance(supplied, list) and session.findings:
            # Findings cite evidence too; the verifier checks those ids as well.
            supplied = list(dict.fromkeys([*supplied, *(value for finding in session.findings
                                                        for value in finding["evidence_ids"])]))
            notes = "; ".join(f"{finding['statement']} [{', '.join(finding['evidence_ids'])}]"
                              for finding in session.findings)
            decision = {**decision, "reasoning_summary": f"{decision.get('reasoning_summary') or ''} "
                                                         f"Findings: {notes}".strip()}
        if not isinstance(verified, list) or not isinstance(supplied, list):
            session.mcp_error = "invalid_agent_decision_shape"
            session.agent_checks.append(Check(
                name="agent decision", status="fail",
                detail="decision JSON lacks verified_claim_ids / additional_evidence_ids lists"))
            return
        unknown_claims = [str(item) for item in verified if item not in session.shown_claim_ids]
        verified = [item for item in verified if item in session.shown_claim_ids]
        session.agent_checks.append(Check(
            name="agent claim IDs",
            status="pass" if not unknown_claims else ("fail" if not verified else "warn"),
            detail=(f"{len(verified)} preliminary claims confirmed" if not unknown_claims else
                    f"{len(unknown_claims)} claim IDs are not in the preliminary graph and were ignored: "
                    + ", ".join(unknown_claims[:5])),
        ))
        unobserved = [str(item) for item in supplied if str(item) not in session.observed]
        cited = [item for item in supplied if str(item) in session.observed]
        if len(cited) > session.evidence_limit:
            session.agent_checks.append(Check(
                name="verification evidence budget", status="info",
                detail=f"checking {session.evidence_limit} of {len(cited)} observed IDs",
            ))
            cited = cited[:session.evidence_limit]
        session.used_for_verification = [str(item) for item in cited]
        session.agent_checks.append(Check(
            name="agent evidence IDs",
            status=("skipped" if not unobserved and not cited else
                    "pass" if not unobserved else "fail" if not cited else "warn"),
            detail=("no additional evidence supplied" if not unobserved and not cited else
                    f"{len(cited)} IDs observed in tool results" if not unobserved else
                    f"{len(unobserved)} IDs were never returned by a tool and were rejected: "
                    + ", ".join(unobserved[:5])),
        ))
        if base_run.status == "incomplete":
            session.agent_checks.append(Check(
                name="independent verification", status="skipped",
                detail="shared research stopped before an answer could be verified",
            ))
        elif verified or cited:
            # A second opinion from a fresh context: does the cited evidence
            # actually support what the agent decided? Advisory, never a gate.
            session.independent, check = self._independent_verification(
                session.question, decision, verified, cited, session.corpus_calls, base_run,
            )
            session.agent_checks.append(check)
        if unknown_claims and not verified and not cited:
            session.mcp_error = "agent_claim_id_not_in_preliminary_graph"
        elif unobserved and not cited and not verified:
            session.mcp_error = "agent_evidence_ids_not_observed:" + ",".join(unobserved[:5])
        elif not session.corpus_calls:
            session.mcp_error = "agent_used_no_corpus_tools"
        elif getattr(self.baseline, "last_route", None) == "verified_facts":
            # The answer was computed from verified corpus facts; the
            # agent's review is recorded but cannot drop correct facts.
            pass
        elif cited:
            # The observed passage IDs were already supplied to the
            # shared pipeline before its first retrieval pass.
            pass
        elif verified and base_run.chosen_reading is not None:
            self._narrow_to_verified_claims(session, verified)
        else:
            session.mcp_error = "agent_verified_no_claims"

    def _narrow_to_verified_claims(self, session: _Session, verified: list) -> None:
        base_run = session.base_run
        assert base_run is not None and session.result is not None
        reading = base_run.readings[base_run.chosen_reading]
        selected = [claim for claim in reading.claims if claim.id in verified]
        keys = {
            (claim.fields["subject"].value, claim.fields["relation"].value,
             claim.fields["object"].value)
            for claim in selected
        }
        triples = [item for item in reading.triples if (item.s, item.p, item.o) in keys]
        if not (selected and triples):
            session.mcp_error = "agent_selected_no_supported_claims"
            return
        answer = "The verified corpus claims report: " + "; ".join(
            f"{claim.fields['object'].value} [{', '.join(claim.source_ids)}]"
            for claim in selected
        ) + "."
        reading = reading.model_copy(update={
            "claims": selected, "triples": triples, "answer": answer,
            "claim_relations": [link for link in reading.claim_relations
                                if link.source_claim in verified and link.target_claim in verified],
        })
        session.base_run = base_run.model_copy(update={"readings": [reading]})
        session.result["artifacts"]["answer.md"] = answer.encode()
        session.result["artifacts"]["claims.json"] = json.dumps({
            "claims": [claim.model_dump(by_alias=True) for claim in selected],
            "relations": [link.model_dump(by_alias=True) for link in reading.claim_relations],
        }).encode()

    def _record_session_error(self, session: _Session) -> None:
        self._tool_sink(session, {
            "call_id": f"tool-{len(session.tool_calls) + 1:03d}",
            "agent": "supervisor",
            "tool": "mcp_session",
            "arguments": {"question": session.question},
            "result": None,
            "result_count": 0,
            "duration_ms": round((monotonic() - session.started) * 1000, 1),
            "status": "deferred" if session.heavy_busy else "error",
            "error": session.mcp_error,
        })

    # ─── Result ──────────────────────────────────────────────────────────────

    def _stage_checks(self, session: _Session) -> list[Check]:
        """One check per agent stage, so "MCP returned nothing", "the agent did not use
        tools" and "the agent's decision failed validation" read differently."""
        base_run, mcp_error, corpus_calls = session.base_run, session.mcp_error, session.corpus_calls
        assert base_run is not None
        checks = list(session.agent_checks)
        if session.heavy_busy:
            research = Check(name="agent research (MCP)", status="skipped", detail=mcp_error or "heavy model busy")
        elif mcp_error and mcp_error.startswith("MCP agent workflow failed"):
            research = Check(name="agent research (MCP)", status="fail", detail=mcp_error)
        else:
            empty = sum(1 for item in corpus_calls if not item.get("result_count"))
            research = Check(name="agent research (MCP)", status="pass" if corpus_calls else "fail",
                             detail=(f"{len(corpus_calls)} corpus tool calls"
                                     + (f", {empty} returned no matches" if empty else ""))
                             if corpus_calls else "the agent answered without calling a corpus tool")
        checks.insert(0, research)
        checks.append(Check(
            name="agentic execution",
            status="pass" if not mcp_error and base_run.status != "incomplete" else
            "fail" if base_run.status == "incomplete" else "warn",
            detail=("agent review accepted" if not mcp_error and base_run.status != "incomplete" else
                    base_run.reason or "pipeline incomplete" if base_run.status == "incomplete" else
                    f"degraded ({mcp_error}); showing the grounded pipeline answer"),
        ))
        if base_run.chosen_reading is not None and base_run.readings:
            measure = base_run.readings[base_run.chosen_reading].measure
            checks.append(Check(
                name="interpretation", status="info",
                detail=f"researched as '{session.concept.label}'" + (f"; answered using: {measure}" if measure else ""),
            ))
        return checks

    def _assemble(self, session: _Session) -> dict[str, Any]:
        base_run, result, mcp_error = session.base_run, session.result, session.mcp_error
        assert base_run is not None and result is not None
        tool_calls = session.tool_calls
        decision = session.agent_decision if isinstance(session.agent_decision, dict) else {}
        plans = [str(decision.get("reasoning_summary"))] if decision.get("reasoning_summary") else []
        plans.extend(f"MCP: {record['tool']}" for record in tool_calls if record.get("tool") != "mcp_session")
        baseline_trace = [
            row.model_copy(update={"at_s": round(row.at_s + session.baseline_started_s, 3)})
            for row in base_run.trace
        ]
        trace = sorted(baseline_trace + session.trace, key=lambda row: row.at_s)
        at_s = (trace[-1].at_s if trace else 0.0) + 0.01
        for plan in plans:
            row = TraceRow(atS=at_s, action="Agent planner lens", returned=plan)
            self.trace_sink(row)
            trace.append(row)
            at_s += 0.01
        ambiguous = base_run.status == "awaiting_reading"
        run = base_run.model_copy(
            update={
                "checks": list(base_run.checks) + self._stage_checks(session),
                "path": "agentic",
                "concept": session.concept,
                "duration_s": round(monotonic() - session.started, 3),
                "plans": plans,
                "trace": trace,
                "readings": list(base_run.readings),
                "model_agent": getattr(self.agent_model, "model", "recorded"),
                "artifacts": list(required_artifacts("agentic")),
                "status": base_run.status,
                "chosen_reading": base_run.chosen_reading,
                "producing_path": "baseline_fallback" if mcp_error or base_run.status == "incomplete" else "agentic",
                "execution_status": "failed" if base_run.status == "incomplete" else
                                    "degraded" if mcp_error else "completed",
                "reason": (
                    base_run.reason if base_run.status in {"incomplete", "corpus_absent", "no_evidence_found"}
                    else f"agentic_degraded: {mcp_error}" if mcp_error
                    else None
                ),
            }
        )
        artifacts = {
            name: result["artifacts"].get(name, b"")
            for name in required_artifacts("agentic")
            if name != "run.json"
        }
        agent_evidence = sorted({
            str(value) for item in tool_calls for value in self._evidence_ids(item.get("result"))
        })
        artifacts["plans.json"] = json.dumps(plans, indent=2).encode()
        artifacts["research_ledger.json"] = json.dumps({
            "question": session.question,
            "tool_calls": [
                {"tool": item["tool"], "purpose": item.get("purpose"),
                 "expected_evidence_type": item.get("expected_evidence_type"),
                 "result_count": item.get("result_count")}
                for item in tool_calls if item["tool"] not in {"context_checkpoint"}
            ],
            "evidence_ids": agent_evidence,
            "findings": session.findings,
        }, indent=2).encode()
        artifacts["context_checkpoints.json"] = json.dumps([
            item.get("result") for item in tool_calls if item["tool"] == "context_checkpoint"
        ], indent=2).encode()
        shared_ledger = json.loads(result["artifacts"].get("ledger.json", b"{}").decode() or "{}")
        shared_ledger.setdefault("evidence_bundle", {}).update({"agent_evidence_ids": agent_evidence})
        shared_ledger["execution_metadata"] = {
            "controller": "agentic",
            "producing_path": run.producing_path,
            "execution_status": run.execution_status,
            "skills_loaded": [item["result"] for item in tool_calls if item["tool"] == "load_skill"],
        }
        artifacts["ledger.json"] = json.dumps(shared_ledger, indent=2).encode()
        artifacts["disagreement.json"] = json.dumps({"independent_plans": len(plans), "agreement": 0.42 if ambiguous else None, "awaiting_reading": ambiguous}, indent=2).encode()
        artifacts["readings.json"] = json.dumps([r.model_dump(by_alias=True) for r in run.readings], indent=2).encode()
        supported = bool(run.readings) and all(
            triple.source and triple.evidence
            for reading in run.readings for triple in reading.triples
        ) and any(reading.triples for reading in run.readings)
        supplied = [str(item) for item in decision.get("additional_evidence_ids") or []]
        artifacts["verification.json"] = json.dumps({
            # Separate checks: provenance of the answer's triples is not the same
            # as a successful agentic execution.
            "triple_provenance": {
                "verified": run.status == "answered" and supported,
                "detail": "every answer triple has a source and evidence" if supported else "not all triples are sourced",
            },
            "agentic_execution": {
                "status": run.execution_status, "producing_path": run.producing_path,
                "reason": mcp_error,
            },
            "agent_evidence_ids": {
                "supplied": supplied,
                "used_for_retrieval": session.selected_ids,
                "findings": session.findings,
                "used_for_verification": session.used_for_verification,
                "observed_in_session": sorted(item for item in supplied if item in session.observed),
                "rejected_unobserved": sorted(item for item in supplied if item not in session.observed),
            },
            "independent_verification": session.independent,
            "checks": [check.model_dump() for check in run.checks],
            "heavy_model_used": not session.heavy_busy and not bool(mcp_error),
            "agentic_degraded": bool(mcp_error),
        }, indent=2).encode()
        artifacts["trace.json"] = json.dumps([row.model_dump(by_alias=True) for row in trace], indent=2).encode()
        baseline_calls = result["artifacts"].get("tool_calls.jsonl", b"")
        agent_calls = b"\n".join(
            json.dumps(item, ensure_ascii=False, default=str).encode() for item in tool_calls
        ) + (b"\n" if tool_calls else b"")
        artifacts["tool_calls.jsonl"] = baseline_calls + agent_calls
        artifacts["agent_draft.md"] = session.agent_final.encode()
        return {"run": run, "artifacts": artifacts}

    def _grounded_context(self, base_run: Run | None, concept: dict | None) -> dict:
        """What the agent knows before research, with optional preliminary claims.

        Claims carry their own fields, so triples are only sent when there are
        no claims. Paper handles are limited to papers the preliminary result
        mentions; the agent reads anything else through the corpus tools.
        """
        reading = (
            base_run.readings[base_run.chosen_reading]
            if base_run is not None and base_run.chosen_reading is not None and base_run.readings else None
        )
        claims = list(reading.claims) if reading else []
        answer_ids = set(reading.answer_claim_ids) if reading else set()
        claims.sort(key=lambda claim: claim.id not in answer_ids)
        mentioned = {claim.paper_id for claim in claims}
        if reading and not claims:
            mentioned |= {triple.s for triple in reading.triples}
        corpus = getattr(self.baseline, "corpus", None)
        papers = []
        for paper_id in sorted(mentioned):
            title = corpus.paper_title(paper_id) if corpus is not None else None
            papers.append({"paper_id": paper_id, "title": title})
        # The passages that grounded the chosen reading: research starts from them.
        by_passage = getattr(corpus, "by_passage", None) or {}
        concept_evidence = [
            {"passage_id": pid, "paper_id": str(by_passage[pid].get("paper_id")),
             "excerpt": " ".join(str(by_passage[pid].get("text", "")).split())[:300]}
            for pid in (concept or {}).get("sample_sources") or [] if pid in by_passage
        ]
        context: dict = {
            "preliminary_answer": reading.answer if reading else None,
            "answer_source": (
                getattr(self.baseline, "last_route", None) if base_run is not None else None
            ) or "research",
            "selected_concept": concept,
            "papers": papers,
            "preliminary_claims": [
                {"id": claim.id, "kind": claim.kind, "paper_id": claim.paper_id,
                 "fields": {key: field.value for key, field in claim.fields.items()},
                 "source_ids": claim.source_ids}
                for claim in claims
            ],
        }
        if concept_evidence:
            context["concept_evidence"] = concept_evidence
        if reading and not claims:
            context["preliminary_triples"] = [
                {"subject": t.s, "predicate": t.p, "object": t.o, "source_id": t.source}
                for t in reading.triples
            ]
        return context


__all__ = ["NON_CORPUS_RECORDS", "AgenticResearch"]
