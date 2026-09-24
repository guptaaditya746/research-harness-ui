"""Live agent-path orchestration over the shared corpus and graph pipeline."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from time import monotonic

import httpx
from harness_baseline import Baseline20
from harness_extraction import model_purpose
from harness_observability import record_tool_call
from harness_schema.models import (
    Check,
    ConceptChoice,
    Metrics,
    Reading,
    Run,
    TraceRow,
    required_artifacts,
)

from .concepts import ConceptProposalError, propose_concepts
from .mcp import MCPToolClient, MCPUnavailable

DEFAULT_CLASSES = {"Paper", "Author", "Venue", "Dataset", "Metric", "Claim"}
# Bookkeeping records in tool_calls that are not corpus research.
NON_CORPUS_RECORDS = {"load_skill", "context_checkpoint", "mcp_session"}


class AgenticResearch:
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
        concept_input = concept.model_dump(by_alias=False)
        tool_calls: list[dict] = []
        started = monotonic()
        trace: list[TraceRow] = []
        mcp_error: str | None = None
        agent_final = ""
        agent_decision: dict = {}
        shown_ids: set[str] = set()
        shown_claim_ids: set[str] = set()
        agent_checks: list[Check] = []
        independent: dict | None = None

        def tool_sink(record: dict) -> None:
            tool_calls.append(record)
            record_tool_call(record, run_id=run_id)
            action = {
                "search_evidence": "Searching corpus evidence",
                "read_evidence": "Inspecting supporting evidence",
                "structured_facts": "Checking canonical metadata",
                "citations": "Checking citation evidence",
                "tables": "Inspecting table evidence",
                "evidence_for_claim": "Verifying claim context",
                "load_skill": "Loading research skill",
                "context_checkpoint": "Context checkpoint",
            }.get(record["tool"], "Researching corpus evidence")
            row = TraceRow(
                atS=round(monotonic() - started, 3),
                action=action,
                returned=(
                    f"{record.get('status')} · {record.get('result_count', 0)} items · "
                    f"{record.get('duration_ms', 0)} ms"
                ),
                purpose=record.get("purpose"),
                expectedEvidenceType=record.get("expected_evidence_type"),
            )
            trace.append(row)
            self.trace_sink(row)

        # The user already chose an agent-proposed concept for this run.
        clarification_needed = False
        queue_depth = None if clarification_needed else self._heavy_queue_depth()
        queue_limit = int(os.getenv("HARNESS_HEAVY_QUEUE_THRESHOLD", "8"))
        # Granite controls MCP research. Qwen queue pressure must not turn an
        # otherwise healthy Granite-led agentic execution into a fallback.
        controller_uses_heavy = getattr(self.agent_model, "model", "") == os.getenv(
            "HARNESS_MODEL_HIGH", "heavy-model"
        )
        heavy_busy = (
            controller_uses_heavy
            and queue_depth is not None
            and queue_depth >= queue_limit
        )
        if heavy_busy:
            self.baseline.refinement_model = None
            mcp_error = f"heavy_model_busy: {queue_depth} requests waiting"
            row = TraceRow(atS=0.0, action="Heavy model deferred", returned=mcp_error)
            trace.append(row)
            self.trace_sink(row)

        # Search through MCP before the shared pipeline verifies its answer.
        # Otherwise a rejected draft prevents the agent from gathering the
        # very evidence that could make the answer supportable.
        if not heavy_busy and not clarification_needed:
            mcp = MCPToolClient(tool_sink, agent="supervisor")
            try:
                grounded_context = self._grounded_context(None, concept_input)
                shown_ids = self._evidence_ids(grounded_context)
                shown_claim_ids = {
                    str(claim["id"]) for claim in grounded_context.get("preliminary_claims", [])
                }
                agent_final = mcp.run_agent(self.agent_model, question, grounded_context)
                if agent_final:
                    try:
                        start, end = agent_final.find("{"), agent_final.rfind("}")
                        agent_decision = json.loads(agent_final[start:end + 1])
                        if not isinstance(agent_decision, dict):
                            raise TypeError("agent decision must be an object")
                    except (ValueError, TypeError):
                        mcp_error = "agent_final_not_structured"
                    row = TraceRow(
                        atS=round(monotonic() - started, 3),
                        action="Agent research decision",
                        returned=agent_final[:500],
                    )
                    trace.append(row)
                    self.trace_sink(row)
            except MCPUnavailable as exc:
                mcp_error = str(exc)
                row = TraceRow(
                    atS=round(monotonic() - started, 3),
                    action="MCP unavailable",
                    returned=str(exc),
                )
                trace.append(row)
                self.trace_sink(row)
        if not clarification_needed and not heavy_busy and not agent_decision and not mcp_error:
            mcp_error = "agent_returned_no_research_decision"
        corpus_calls = [item for item in tool_calls if item["tool"] not in NON_CORPUS_RECORDS]
        # Only IDs the agent actually saw in this session (tool results and the
        # preliminary result shown to it) count as evidence; existing somewhere
        # in the corpus is not enough.
        observed = shown_ids.union(*(self._evidence_ids(item.get("result")) for item in corpus_calls))
        if mcp_error and corpus_calls:
            # A failed controller decision stays visible as degraded. Recover
            # a small set of actual passage handles from successful tool calls
            # so a timeout or malformed final answer does not waste research
            # already gathered before schema inference.
            recovered: list[str] = []

            def passage_ids(value: object) -> list[str]:
                if isinstance(value, dict):
                    found = [str(value["passage_id"])] if value.get("passage_id") else []
                    return found + [handle for child in value.values() for handle in passage_ids(child)]
                if isinstance(value, list):
                    return [handle for child in value for handle in passage_ids(child)]
                return []

            for call in corpus_calls:
                if call.get("status") != "ok" or call.get("tool") not in {
                    "search_evidence", "read_evidence", "rank_evidence", "passages",
                }:
                    continue
                recovered.extend(passage_ids(call.get("result"))[:2])
                recovered = list(dict.fromkeys(recovered))[:12]
                if len(recovered) == 12:
                    break
            if recovered:
                agent_decision = {
                    "verified_claim_ids": [],
                    "additional_evidence_ids": recovered,
                    "reasoning_summary": "Recovered observed MCP passages after the controller failed.",
                }
                agent_checks.append(Check(
                    name="agent passage recovery", status="warn",
                    detail=f"{len(recovered)} observed passage IDs recovered from MCP tool results",
                ))
        supplied_ids = agent_decision.get("additional_evidence_ids")
        evidence_limit = max(1, int(os.getenv("HARNESS_AGENTIC_MAX_MCP_SEEDS", "6")))
        selected_ids = (
            [str(item) for item in supplied_ids if str(item) in observed]
            if isinstance(supplied_ids, list) else []
        )
        selected_ids = list(dict.fromkeys(selected_ids))
        corpus_passages = getattr(getattr(self.baseline, "corpus", None), "by_passage", None)
        if corpus_passages is not None:
            selected_ids = [item for item in selected_ids if item in corpus_passages]
            diverse = []
            seen_papers = set()
            for source_id in selected_ids:
                paper_id = str(corpus_passages[source_id]["paper_id"])
                if paper_id not in seen_papers:
                    diverse.append(source_id)
                    seen_papers.add(paper_id)
            selected_ids = list(dict.fromkeys([*diverse, *selected_ids]))
        deferred_ids = max(0, len(selected_ids) - evidence_limit)
        selected_ids = selected_ids[:evidence_limit]
        if deferred_ids:
            agent_checks.append(Check(
                name="agent evidence budget", status="info",
                detail=f"{len(selected_ids)} MCP passages selected; {deferred_ids} deferred for response time",
            ))
        agent_checks.append(Check(
            name="agent passage seeds",
            status="pass" if selected_ids else "skipped",
            detail=f"{len(selected_ids)} MCP passage IDs supplied before retrieval",
        ))
        previous_seed_ids = list(getattr(self.baseline, "seed_passage_ids", []))
        if hasattr(self.baseline, "seed_passage_ids"):
            self.baseline.seed_passage_ids = list(dict.fromkeys([*previous_seed_ids, *selected_ids]))
        baseline_started_s = monotonic() - started
        original_trace_sink = getattr(self.baseline, "trace_sink", None)
        if callable(original_trace_sink):
            self.baseline.trace_sink = lambda row: original_trace_sink(
                row.model_copy(update={"at_s": round(row.at_s + baseline_started_s, 3)})
            )
        try:
            result = self.baseline.run(question, budget, run_id, concept=concept_input)
        finally:
            if hasattr(self.baseline, "seed_passage_ids"):
                self.baseline.seed_passage_ids = previous_seed_ids
            if callable(original_trace_sink):
                self.baseline.trace_sink = original_trace_sink
        base_run = result["run"]
        used_for_verification: list[str] = []
        if agent_decision and not mcp_error:
            available = shown_claim_ids
            verified = agent_decision.get("verified_claim_ids")
            evidence_ids = agent_decision.get("additional_evidence_ids")
            if not isinstance(verified, list) or not isinstance(evidence_ids, list):
                mcp_error = "invalid_agent_decision_shape"
                agent_checks.append(Check(name="agent decision", status="fail",
                                          detail="decision JSON lacks verified_claim_ids / additional_evidence_ids lists"))
            else:
                unknown_claims = [str(item) for item in verified if item not in available]
                verified = [item for item in verified if item in available]
                agent_checks.append(Check(
                    name="agent claim IDs",
                    status="pass" if not unknown_claims else ("fail" if not verified else "warn"),
                    detail=(f"{len(verified)} preliminary claims confirmed" if not unknown_claims else
                            f"{len(unknown_claims)} claim IDs are not in the preliminary graph and were ignored: "
                            + ", ".join(unknown_claims[:5])),
                ))
                unobserved = [str(item) for item in evidence_ids if str(item) not in observed]
                evidence_ids = [item for item in evidence_ids if str(item) in observed]
                if len(evidence_ids) > evidence_limit:
                    agent_checks.append(Check(
                        name="verification evidence budget", status="info",
                        detail=f"checking {evidence_limit} of {len(evidence_ids)} observed IDs",
                    ))
                    evidence_ids = evidence_ids[:evidence_limit]
                used_for_verification = [str(item) for item in evidence_ids]
                agent_checks.append(Check(
                    name="agent evidence IDs",
                    status=("skipped" if not unobserved and not evidence_ids else
                            "pass" if not unobserved else "fail" if not evidence_ids else "warn"),
                    detail=("no additional evidence supplied" if not unobserved and not evidence_ids else
                            f"{len(evidence_ids)} IDs observed in tool results" if not unobserved else
                            f"{len(unobserved)} IDs were never returned by a tool and were rejected: "
                            + ", ".join(unobserved[:5])),
                ))
                if base_run.status == "incomplete":
                    agent_checks.append(Check(
                        name="independent verification", status="skipped",
                        detail="shared research stopped before an answer could be verified",
                    ))
                elif verified or evidence_ids:
                    # A second opinion from a fresh context: does the cited evidence
                    # actually support what the agent decided? Advisory, never a gate.
                    independent, check = self._independent_verification(
                        question, agent_decision, verified, evidence_ids, corpus_calls, base_run,
                    )
                    agent_checks.append(check)
                if unknown_claims and not verified and not evidence_ids:
                    mcp_error = "agent_claim_id_not_in_preliminary_graph"
                elif unobserved and not evidence_ids and not verified:
                    mcp_error = "agent_evidence_ids_not_observed:" + ",".join(unobserved[:5])
                elif not corpus_calls:
                    mcp_error = "agent_used_no_corpus_tools"
                elif getattr(self.baseline, "last_route", None) == "verified_facts":
                    # The answer was computed from verified corpus facts; the
                    # agent's review is recorded but cannot drop correct facts.
                    pass
                elif evidence_ids:
                    # The observed passage IDs were already supplied to the
                    # shared pipeline before its first retrieval pass.
                    pass
                elif verified and base_run.chosen_reading is not None:
                    reading = base_run.readings[base_run.chosen_reading]
                    selected = [claim for claim in reading.claims if claim.id in verified]
                    keys = {
                        (claim.fields["subject"].value, claim.fields["relation"].value,
                         claim.fields["object"].value)
                        for claim in selected
                    }
                    triples = [item for item in reading.triples if (item.s, item.p, item.o) in keys]
                    if selected and triples:
                        answer = "The verified corpus claims report: " + "; ".join(
                            f"{claim.fields['object'].value} [{', '.join(claim.source_ids)}]"
                            for claim in selected
                        ) + "."
                        reading = reading.model_copy(update={
                            "claims": selected, "triples": triples, "answer": answer,
                            "claim_relations": [link for link in reading.claim_relations
                                                if link.source_claim in verified and link.target_claim in verified],
                        })
                        base_run = base_run.model_copy(update={"readings": [reading]})
                        result["artifacts"]["answer.md"] = answer.encode()
                        result["artifacts"]["claims.json"] = json.dumps({
                            "claims": [claim.model_dump(by_alias=True) for claim in selected],
                            "relations": [link.model_dump(by_alias=True) for link in reading.claim_relations],
                        }).encode()
                    else:
                        mcp_error = "agent_selected_no_supported_claims"
                else:
                    mcp_error = "agent_verified_no_claims"
        if mcp_error:
            tool_sink({
                "call_id": f"tool-{len(tool_calls) + 1:03d}",
                "agent": "supervisor",
                "tool": "mcp_session",
                "arguments": {"question": question},
                "result": None,
                "result_count": 0,
                "duration_ms": round((monotonic() - started) * 1000, 1),
                "status": "deferred" if heavy_busy else "error",
                "error": mcp_error,
            })

        plans = [str(agent_decision.get("reasoning_summary"))] if agent_decision.get("reasoning_summary") else []
        plans.extend(f"MCP: {record['tool']}" for record in tool_calls if record.get("tool") != "mcp_session")
        baseline_trace = [
            row.model_copy(update={"at_s": round(row.at_s + baseline_started_s, 3)})
            for row in base_run.trace
        ]
        trace = sorted(baseline_trace + trace, key=lambda row: row.at_s)
        at_s = (trace[-1].at_s if trace else 0.0) + 0.01
        for plan in plans:
            row = TraceRow(atS=at_s, action="Agent planner lens", returned=plan)
            self.trace_sink(row)
            trace.append(row)
            at_s += 0.01
        ambiguous = base_run.status == "awaiting_reading"
        readings = list(base_run.readings)
        status = base_run.status
        chosen = base_run.chosen_reading
        # One check per agent stage, so "MCP returned nothing", "the agent did
        # not use tools" and "the agent's decision failed validation" read differently.
        if heavy_busy:
            research = Check(name="agent research (MCP)", status="skipped", detail=mcp_error or "heavy model busy")
        elif mcp_error and mcp_error.startswith("MCP agent workflow failed"):
            research = Check(name="agent research (MCP)", status="fail", detail=mcp_error)
        else:
            empty = sum(1 for item in corpus_calls if not item.get("result_count"))
            research = Check(name="agent research (MCP)", status="pass" if corpus_calls else "fail",
                             detail=(f"{len(corpus_calls)} corpus tool calls"
                                     + (f", {empty} returned no matches" if empty else ""))
                             if corpus_calls else "the agent answered without calling a corpus tool")
        agent_checks.insert(0, research)
        agent_checks.append(Check(
            name="agentic execution",
            status="pass" if not mcp_error and base_run.status != "incomplete" else
            "fail" if base_run.status == "incomplete" else "warn",
            detail=("agent review accepted" if not mcp_error and base_run.status != "incomplete" else
                    base_run.reason or "pipeline incomplete" if base_run.status == "incomplete" else
                    f"degraded ({mcp_error}); showing the grounded pipeline answer"),
        ))
        if concept is not None and base_run.chosen_reading is not None and base_run.readings:
            measure = base_run.readings[base_run.chosen_reading].measure
            agent_checks.append(Check(
                name="interpretation", status="info",
                detail=f"researched as '{concept.label}'" + (f"; answered using: {measure}" if measure else ""),
            ))
        run = base_run.model_copy(
            update={
                "checks": list(base_run.checks) + agent_checks,
                "path": "agentic",
                "concept": concept,
                "duration_s": round(monotonic() - started, 3),
                "plans": plans,
                "trace": trace,
                "readings": readings,
                "model_agent": getattr(self.agent_model, "model", "recorded"),
                "artifacts": list(required_artifacts("agentic")),
                "status": status,
                "chosen_reading": chosen,
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
        artifacts["plans.json"] = json.dumps(plans, indent=2).encode()
        artifacts["research_ledger.json"] = json.dumps({
            "question": question,
            "tool_calls": [
                {"tool": item["tool"], "purpose": item.get("purpose"),
                 "expected_evidence_type": item.get("expected_evidence_type"),
                 "result_count": item.get("result_count")}
                for item in tool_calls if item["tool"] not in {"context_checkpoint"}
            ],
            "evidence_ids": sorted({
                str(value) for item in tool_calls for value in self._evidence_ids(item.get("result"))
            }),
        }, indent=2).encode()
        artifacts["context_checkpoints.json"] = json.dumps([
            item.get("result") for item in tool_calls if item["tool"] == "context_checkpoint"
        ], indent=2).encode()
        shared_ledger = json.loads(result["artifacts"].get("ledger.json", b"{}").decode() or "{}")
        shared_ledger.setdefault("evidence_bundle", {}).update({
            "agent_evidence_ids": sorted({
                str(value) for item in tool_calls for value in self._evidence_ids(item.get("result"))
            }),
        })
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
        decision = agent_decision if isinstance(agent_decision, dict) else {}
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
                "supplied": [str(item) for item in decision.get("additional_evidence_ids") or []],
                "used_for_retrieval": selected_ids,
                "used_for_verification": used_for_verification,
                "observed_in_session": sorted(
                    str(item) for item in decision.get("additional_evidence_ids") or [] if str(item) in observed),
                "rejected_unobserved": sorted(
                    str(item) for item in decision.get("additional_evidence_ids") or [] if str(item) not in observed),
            },
            "independent_verification": independent,
            "checks": [check.model_dump() for check in run.checks],
            "heavy_model_used": not heavy_busy and not bool(mcp_error),
            "agentic_degraded": bool(mcp_error),
        }, indent=2).encode()
        artifacts["trace.json"] = json.dumps([row.model_dump(by_alias=True) for row in trace], indent=2).encode()
        baseline_calls = result["artifacts"].get("tool_calls.jsonl", b"")
        agent_calls = b"\n".join(
            json.dumps(item, ensure_ascii=False, default=str).encode() for item in tool_calls
        ) + (b"\n" if tool_calls else b"")
        artifacts["tool_calls.jsonl"] = baseline_calls + agent_calls
        artifacts["agent_draft.md"] = agent_final.encode()
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
        if reading and not claims:
            context["preliminary_triples"] = [
                {"subject": t.s, "predicate": t.p, "object": t.o, "source_id": t.source}
                for t in reading.triples
            ]
        return context

    def _allowed_classes(self) -> set[str]:
        extraction = getattr(self.baseline, "extraction", None)
        names = set(getattr(extraction, "class_names", {}) or {})
        return names or set(DEFAULT_CLASSES)

    def propose(self, question: str, budget: str, run_id: str) -> dict:
        """Let the agent model propose 3-4 concept readings before research."""
        started = monotonic()
        started_at = datetime.now(UTC)
        trace: list[TraceRow] = []

        def emit(action: str, returned: str) -> None:
            row = TraceRow(atS=round(monotonic() - started, 3), action=action, returned=returned)
            trace.append(row)
            self.trace_sink(row)

        emit("Agent proposing concepts", "Interpreting the question before research")
        model_name = getattr(self.agent_model, "model", "recorded")
        try:
            concepts = propose_concepts(
                self.high_model, question, self._allowed_classes(),
                corpus=getattr(self.baseline, "corpus", None),
            )
        except ConceptProposalError as exc:
            emit("Concept proposal failed", str(exc))
            run = Run(
                id=run_id, startedAt=started_at, durationS=round(monotonic() - started, 3),
                question=question, path="agentic", budget=budget, status="incomplete",
                reason=f"concept_proposal_failed: {exc}"[:300], trace=trace,
                modelAgent=model_name, artifacts=list(required_artifacts("agentic")),
            )
            artifacts = {name: b"" for name in required_artifacts("agentic") if name != "run.json"}
            artifacts["answer.md"] = b"The agent could not propose concept readings for this question."
            artifacts["trace.json"] = json.dumps(
                [row.model_dump(by_alias=True) for row in trace], indent=2
            ).encode()
            return {"run": run, "artifacts": artifacts}

        for index, item in enumerate(concepts, start=1):
            emit(
                "Agent concept",
                f"{index}. {item.label} · {item.evidence_hits} corpus passages",
            )
        readings = [
            Reading(
                label=item.label, gloss=item.gloss, classes=item.classes, axioms=[],
                sparql="", answer="", triples=[], metrics=Metrics(), status="proposed",
            )
            for item in concepts
        ]
        run = Run(
            id=run_id, startedAt=started_at, durationS=round(monotonic() - started, 3),
            question=question, path="agentic", budget=budget, status="awaiting_reading",
            readings=readings, chosenReading=None, trace=trace,
            plans=[f"{item.label}: {item.gloss}" for item in concepts],
            concepts=[
                ConceptChoice(
                    label=item.label, gloss=item.gloss, classes=item.classes,
                    searchTerms=item.search_terms, evidenceHits=item.evidence_hits,
                )
                for item in concepts
            ],
            modelAgent=model_name, artifacts=list(required_artifacts("agentic")),
        )
        artifacts = {name: b"" for name in required_artifacts("agentic") if name != "run.json"}
        artifacts["plans.json"] = json.dumps(
            [item.as_dict() for item in concepts], indent=2
        ).encode()
        artifacts["readings.json"] = json.dumps(
            [reading.model_dump(by_alias=True) for reading in readings], indent=2
        ).encode()
        artifacts["disagreement.json"] = json.dumps(
            {"proposed_concepts": len(concepts), "awaiting_reading": True}, indent=2
        ).encode()
        artifacts["answer.md"] = b"Choose one of the agent's concept readings to continue."
        artifacts["trace.json"] = json.dumps(
            [row.model_dump(by_alias=True) for row in trace], indent=2
        ).encode()
        return {"run": run, "artifacts": artifacts}

    @staticmethod
    def _heavy_queue_depth() -> int | None:
        """Read vLLM queue pressure when the configured high model exposes metrics."""
        base = os.getenv("VLLM_HEAVY_API_BASE", "").rstrip("/")
        if not base:
            return None
        endpoint = base.removesuffix("/v1") + "/metrics"
        try:
            response = httpx.get(
                endpoint,
                headers={"Authorization": f"Bearer {os.getenv('VLLM_API_KEY', 'local')}"},
                timeout=3,
            )
            response.raise_for_status()
            for line in response.text.splitlines():
                if line.startswith("vllm:num_requests_waiting{"):
                    return int(float(line.rsplit(" ", 1)[-1]))
        except (httpx.HTTPError, ValueError):
            return None
        return None

    def _independent_verification(
        self, question: str, decision: dict, verified: list, evidence_ids: list,
        corpus_calls: list[dict], base_run: Run,
    ) -> tuple[dict | None, Check]:
        """One structured call, in a fresh context, judging whether the cited evidence supports the decision.

        The verifier never sees the agent's conversation: only the question, the
        decision and the text of the evidence items it cites (looked up in this
        session's tool results and preliminary claims). It cannot add evidence,
        only mark cited items supported or not, and it never fails a run.
        """
        verifier = getattr(self.baseline, "refinement_model", None) or self.agent_model
        complete = getattr(verifier, "complete_json", None)
        if complete is None:
            return None, Check(name="independent verification", status="skipped", detail="no verifier model configured")
        wanted = {str(item) for item in evidence_ids}
        excerpts: dict[str, str] = {}

        def collect(value: object) -> None:
            if isinstance(value, dict):
                handle = next((str(value[key]) for key in ("passage_id", "fact_id", "reference_id", "source_id")
                               if value.get(key) not in (None, "")), None)
                if handle in wanted and handle not in excerpts:
                    if {"subject", "predicate", "object"} <= value.keys():
                        excerpts[handle] = f"{value['subject']} {value['predicate']} {value['object']}"
                    else:
                        text = next((str(value[key]) for key in ("text", "evidence", "snippet", "title") if value.get(key)), "")
                        excerpts[handle] = text[:600] if text else json.dumps(value, ensure_ascii=False, default=str)[:600]
                for item in value.values():
                    collect(item)
            elif isinstance(value, list):
                for item in value:
                    collect(item)

        for call in corpus_calls:
            collect(call.get("result"))
        claims = {claim.id: claim for reading in base_run.readings for claim in reading.claims}
        cited_claims = [{
            "id": claim_id,
            "fields": {key: field.value for key, field in claims[claim_id].fields.items()},
            "evidence": {source: text[:400] for source, text in claims[claim_id].evidence.items()},
        } for claim_id in verified if claim_id in claims]
        request = {
            "question": question,
            "agent_decision": {
                "reasoning_summary": str(decision.get("reasoning_summary") or "")[:1200],
                "verified_claim_ids": [str(item) for item in verified],
                "additional_evidence_ids": sorted(wanted),
            },
            "cited_claims": cited_claims,
            "cited_evidence": excerpts,
            "evidence_not_found_in_session": sorted(wanted - set(excerpts)),
        }
        system = (
            "You are an independent verifier for a research harness over a fixed corpus. You see only "
            "the question, an agent's decision, and the text of the evidence it cites. For each cited "
            "claim id and evidence id, decide whether the cited text supports what the decision says "
            "about the question. You cannot add evidence or ids. Return strict JSON: "
            '{"supported_ids": [ids whose text supports the decision], "unsupported_ids": [ids whose text '
            'does not, or is missing], "notes": one sentence}.'
        )
        try:
            with model_purpose("independent verification"):
                payload = complete(system, json.dumps(request, ensure_ascii=False))
        except Exception as exc:  # noqa: BLE001 - advisory step; report, never fail the run
            return ({"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"[:200]},
                    Check(name="independent verification", status="skipped",
                          detail=f"verifier call failed: {type(exc).__name__}"))
        cited = set(excerpts) | {claim["id"] for claim in cited_claims}
        supported = [str(item) for item in (payload.get("supported_ids") or []) if str(item) in cited] \
            if isinstance(payload, dict) else []
        unsupported = [str(item) for item in (payload.get("unsupported_ids") or []) if str(item) in cited] \
            if isinstance(payload, dict) else []
        unsupported += sorted((wanted - set(excerpts)) - set(unsupported))
        notes = str(payload.get("notes") or "")[:300] if isinstance(payload, dict) else ""
        result = {"status": "checked", "model": getattr(verifier, "model", "verifier"), "cited": sorted(cited),
                  "supported_ids": supported, "unsupported_ids": unsupported, "notes": notes}
        if not cited:
            status, detail = "skipped", "the decision cites nothing this session returned"
        elif unsupported:
            status = "warn"
            detail = (f"{len(supported)} of {len(cited)} cited items judged to support the decision; "
                      f"not supported: {', '.join(unsupported[:5])}" + (f" — {notes}" if notes else ""))
        else:
            status, detail = "pass", f"all {len(cited)} cited items judged to support the decision" + (f" — {notes}" if notes else "")
        return result, Check(name="independent verification", status=status, detail=detail)

    @staticmethod
    def _evidence_ids(value: object) -> set[str]:
        if isinstance(value, dict):
            found = {
                str(item) for key, item in value.items()
                if key.endswith("_id") and isinstance(item, (str, int))
            }
            for key, item in value.items():
                if key.endswith("_ids") and isinstance(item, list):
                    found.update(str(entry) for entry in item if isinstance(entry, (str, int)))
            for item in value.values():
                found.update(AgenticResearch._evidence_ids(item))
            return found
        if isinstance(value, list):
            return set().union(*(AgenticResearch._evidence_ids(item) for item in value)) if value else set()
        return set()

    @staticmethod
    def _known_evidence_ids(corpus: object) -> set[str]:
        """Accept scientific evidence IDs, not only full-text passage IDs."""
        ids = set(getattr(corpus, "by_passage", {}))
        for row in getattr(corpus, "fact_rows", []):
            if row.get("source_id"):
                ids.add(str(row["source_id"]))
            if row.get("fact_id"):
                ids.add(str(row["fact_id"]))
        for row in getattr(corpus, "references", []):
            if row.get("reference_id"):
                ids.add(str(row["reference_id"]))
        return ids


__all__ = ["AgenticResearch"]
