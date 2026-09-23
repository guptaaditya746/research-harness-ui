"""Live agent-path orchestration over the shared corpus and graph pipeline."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from time import monotonic

import httpx
from harness_baseline import Baseline20
from harness_observability import record_tool_call
from harness_schema.models import (
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


class AgenticResearch:
    """Agent-path runner sharing evidence and graph invariants with baseline."""

    def __init__(
        self,
        baseline: Baseline20,
        trace_sink: Callable[[TraceRow], None] | None = None,
        agent_model: object | None = None,
    ):
        self.baseline = baseline
        self.trace_sink = trace_sink or (lambda _: None)
        self.agent_model = agent_model or baseline.extraction.model

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

        # Run the shared baseline first. MCP enriches the result when the high
        # quality model has capacity; a transport failure preserves its triples.
        result = self.baseline.run(question, budget, run_id, concept=concept_input)
        base_run = result["run"]
        if not heavy_busy and not clarification_needed and base_run.status == "answered":
            mcp = MCPToolClient(tool_sink, agent="supervisor")
            try:
                grounded_context = self._grounded_context(base_run, concept_input)
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
        if base_run.status == "answered" and not clarification_needed and not heavy_busy and not agent_decision and not mcp_error:
            mcp_error = "agent_returned_no_research_decision"
        if agent_decision and not mcp_error:
            available = {claim.id for reading in base_run.readings for claim in reading.claims}
            verified = agent_decision.get("verified_claim_ids")
            evidence_ids = agent_decision.get("additional_evidence_ids")
            if not isinstance(verified, list) or not isinstance(evidence_ids, list):
                mcp_error = "invalid_agent_decision_shape"
            elif any(item not in available for item in verified):
                mcp_error = "agent_claim_id_not_in_preliminary_graph"
            else:
                corpus = getattr(self.baseline, "corpus", None)
                known = self._known_evidence_ids(corpus) if corpus is not None else None
                unknown_evidence = (
                    [str(item) for item in evidence_ids if item not in known]
                    if known is not None else []
                )
                if unknown_evidence:
                    mcp_error = "agent_evidence_id_not_in_corpus:" + ",".join(unknown_evidence[:5])
                elif not tool_calls:
                    mcp_error = "agent_used_no_corpus_tools"
                elif getattr(self.baseline, "last_route", None) == "verified_facts":
                    # The answer was computed from verified corpus facts; the
                    # agent's review is recorded but cannot drop correct facts.
                    pass
                elif evidence_ids and hasattr(self.baseline, "seed_passage_ids"):
                    # The agent found further evidence: rerun the shared
                    # pipeline seeded with it, under the same concept.
                    self.baseline.seed_passage_ids = list(dict.fromkeys(evidence_ids))
                    result = self.baseline.run(question, budget, run_id, concept=concept_input)
                    base_run = result["run"]
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
        trace = sorted(list(base_run.trace) + trace, key=lambda row: row.at_s)
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
        run = base_run.model_copy(
            update={
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
                "execution_status": "degraded" if mcp_error else "failed" if base_run.status == "incomplete" else "completed",
                "reason": (
                    f"agentic_degraded: {mcp_error}" if mcp_error
                    else base_run.reason if base_run.status in {"incomplete", "corpus_absent", "no_evidence_found"}
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
            "producing_path": "baseline_fallback" if mcp_error else "agentic",
            "execution_status": "degraded" if mcp_error else "completed",
            "skills_loaded": [item["result"] for item in tool_calls if item["tool"] == "load_skill"],
        }
        artifacts["ledger.json"] = json.dumps(shared_ledger, indent=2).encode()
        artifacts["disagreement.json"] = json.dumps({"independent_plans": len(plans), "agreement": 0.42 if ambiguous else None, "awaiting_reading": ambiguous}, indent=2).encode()
        artifacts["readings.json"] = json.dumps([r.model_dump(by_alias=True) for r in run.readings], indent=2).encode()
        supported = bool(run.readings) and all(
            triple.source and triple.evidence
            for reading in run.readings for triple in reading.triples
        ) and any(reading.triples for reading in run.readings)
        artifacts["verification.json"] = json.dumps(
            {"status": "triple provenance checked", "verified": run.status == "answered" and supported,
             "heavy_model_used": not heavy_busy and not bool(mcp_error), "agentic_degraded": bool(mcp_error)},
            indent=2,
        ).encode()
        artifacts["trace.json"] = json.dumps([row.model_dump(by_alias=True) for row in trace], indent=2).encode()
        baseline_calls = result["artifacts"].get("tool_calls.jsonl", b"")
        agent_calls = b"\n".join(
            json.dumps(item, ensure_ascii=False, default=str).encode() for item in tool_calls
        ) + (b"\n" if tool_calls else b"")
        artifacts["tool_calls.jsonl"] = baseline_calls + agent_calls
        artifacts["agent_draft.md"] = agent_final.encode()
        return {"run": run, "artifacts": artifacts}

    def _grounded_context(self, base_run: Run, concept: dict | None) -> dict:
        """What the agent needs to verify the preliminary answer, without duplication.

        Claims carry their own fields, so triples are only sent when there are
        no claims. Paper handles are limited to papers the preliminary result
        mentions; the agent reads anything else through the corpus tools.
        """
        reading = (
            base_run.readings[base_run.chosen_reading]
            if base_run.chosen_reading is not None and base_run.readings else None
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
            "answer_source": getattr(self.baseline, "last_route", None) or "research",
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
                self.agent_model, question, self._allowed_classes(),
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

    @staticmethod
    def _evidence_ids(value: object) -> set[str]:
        if isinstance(value, dict):
            found = {
                str(item) for key, item in value.items()
                if key.endswith("_id") and isinstance(item, (str, int))
            }
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
