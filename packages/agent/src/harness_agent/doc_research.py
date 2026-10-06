"""The Agentic path when the evidence document D is the run's input (docs/RUN_INPUT.md). A mixin of
``AgenticResearch``.

- Readings: D's interpretations are the concepts offered (no model call); a reading restricts the run to
  that interpretation's triples.
- Research: the supervisor works over D with the doc_ tools only (``doc_tools.EvidenceDocSession``), records
  findings and names the triples that answer.
- Hand-off: the triples it named (or, when it named none, all of D's, and the run is marked degraded) go to
  the shared pipeline's evidence-document branch, which checks them against their passages and writes the
  answer, with the agent's findings as notes.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from time import monotonic
from typing import Any

from harness_baseline.evidence_doc import EvidenceDoc
from harness_schema.models import (
    Check,
    ConceptChoice,
    Metrics,
    Reading,
    Run,
    TraceRow,
    required_artifacts,
)

from .decisions import id_list
from .doc_tools import EvidenceDocSession, chosen_triples
from .mcp import MCPToolClient, MCPUnavailable


class DocResearch:
    evidence_doc: EvidenceDoc | None

    # ─── Readings ────────────────────────────────────────────────────────────

    def _doc_concepts(self) -> list[ConceptChoice]:
        doc = self.evidence_doc
        assert doc is not None
        return [ConceptChoice(label=item["label"] or item["id"], gloss=item["gloss"] or doc.question,
                              classes=["Paper"], evidenceHits=sum(t.reading == item["id"] for t in doc.triples))
                for item in doc.interpretations]

    def _doc_reading(self, concept: ConceptChoice) -> str | None:
        """The interpretation of D a chosen reading names (by its label), else None (all of D)."""
        doc = self.evidence_doc
        assert doc is not None
        return next((item["id"] for item in doc.interpretations if (item["label"] or item["id"]) == concept.label),
                    None)

    def _propose_from_doc(self, question: str, budget: str, run_id: str) -> dict[str, Any]:
        started_at, started = datetime.now(UTC), monotonic()
        concepts = self._doc_concepts()
        trace = [TraceRow(atS=0.0, action="Readings from the evidence document",
                          returned=f"{len(concepts)} interpretation(s): " + "; ".join(c.label for c in concepts))]
        self.trace_sink(trace[0])
        readings = [Reading(label=c.label, gloss=c.gloss, classes=c.classes, axioms=[], sparql="", answer="",
                            triples=[], metrics=Metrics(), status="proposed") for c in concepts]
        run = Run(id=run_id, startedAt=started_at, durationS=round(monotonic() - started, 3), question=question,
                  path="agentic", budget=budget, status="awaiting_reading", readings=readings, chosenReading=None,
                  trace=trace, plans=[f"{c.label}: {c.gloss}" for c in concepts], concepts=concepts,
                  runInput="evidence_doc", evidenceDoc=self.evidence_doc.summary(),
                  modelAgent=getattr(self.agent_model, "model", "recorded"),
                  artifacts=list(required_artifacts("agentic")))
        artifacts = {name: b"" for name in required_artifacts("agentic") if name != "run.json"}
        artifacts["plans.json"] = json.dumps([c.model_dump(by_alias=True) for c in concepts], indent=2).encode()
        artifacts["answer.md"] = b"Choose one of the evidence document's interpretations to continue."
        artifacts["trace.json"] = json.dumps([row.model_dump(by_alias=True) for row in trace], indent=2).encode()
        return {"run": run, "artifacts": artifacts}

    # ─── Research over D ─────────────────────────────────────────────────────

    def _run_on_doc(self, session: Any) -> dict[str, Any]:
        doc = self.evidence_doc.for_reading(self._doc_reading(session.concept))
        mcp = MCPToolClient(lambda record: self._tool_sink(session, record), agent="supervisor")
        session.notes = mcp.notes
        grounded = {
            "answer_source": "evidence_document", "selected_concept": session.concept_input,
            "evidence_document": {"question_id": doc.question_id, "triples": len(doc.triples),
                                  "interpretation": doc.reading or "all"},
        }
        try:
            session.agent_final = mcp.run_agent(self.agent_model, session.question, grounded,
                                                session=EvidenceDocSession(doc))
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
        self._collect_findings(session)
        named = {value for finding in session.findings for value in finding["evidence_ids"]}
        named |= {value for value in id_list(session.agent_decision.get("additional_evidence_ids"))
                  if value in session.observed}
        chosen = chosen_triples(doc, named)
        if chosen:
            given = doc.subset(chosen)
            session.agent_checks.append(Check(name="agent choice", status="pass",
                                              detail=f"the agent chose {len(chosen)} of {len(doc.triples)} triples of D"))
            if session.mcp_error == "agent_final_not_structured":
                session.mcp_error = None        # its findings carry the choice
        else:
            given = doc
            session.mcp_error = session.mcp_error or "agent_chose_no_triple"
            session.agent_checks.append(Check(name="agent choice", status="warn",
                                              detail=f"the agent named no triple; all {len(doc.triples)} triples of D "
                                                     "went to the evidence review"))
        previous = self.baseline.evidence_doc
        self.baseline.evidence_doc = given
        try:
            self._run_shared_pipeline(session)
        finally:
            self.baseline.evidence_doc = previous
        if session.mcp_error:
            self._record_session_error(session)
        return self._assemble(session)


__all__ = ["DocResearch"]
