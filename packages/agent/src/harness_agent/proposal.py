"""Concept proposal: the agent offers readings of the question before research (one unless it is ambiguous)."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from time import monotonic
from typing import Any

from harness_schema.models import ConceptChoice, Metrics, Reading, Run, TraceRow, required_artifacts

from .concepts import ConceptProposalError, propose_grounded

DEFAULT_CLASSES = {"Paper", "Author", "Venue", "Dataset", "Metric", "Claim"}


class ConceptProposalStep:
    """Mixin of ``AgenticResearch``: the run stops at ``awaiting_reading`` with the proposals."""

    baseline: Any
    trace_sink: Callable[[TraceRow], None]
    agent_model: Any
    high_model: Any

    def _allowed_classes(self) -> set[str]:
        extraction = getattr(self.baseline, "extraction", None)
        names = set(getattr(extraction, "class_names", {}) or {})
        return names or set(DEFAULT_CLASSES)

    def propose(self, question: str, budget: str, run_id: str) -> dict:
        """Let the agent model propose concept readings before research (with the evidence document as
        input: its interpretations, see doc_research.py)."""
        if getattr(self, "evidence_doc", None) is not None:
            return self._propose_from_doc(question, budget, run_id)
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
            proposal = propose_grounded(
                self.high_model, question, self._allowed_classes(),
                corpus=getattr(self.baseline, "corpus", None),
            )
            concepts = proposal.concepts
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

        if proposal.evidence:
            emit("Retrieved evidence before readings",
                 f"{len(proposal.evidence)} passages from "
                 f"{len({row['paper_id'] for row in proposal.evidence})} papers · "
                 f"{len(proposal.keyphrases)} keyphrases")
        for index, item in enumerate(concepts, start=1):
            emit(
                "Agent concept",
                f"{index}. {item.label} · {item.evidence_hits} supporting passages",
            )
        if proposal.hidden:
            emit("Readings hidden", "no supporting passage: " + "; ".join(proposal.hidden))
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
                    sampleSources=item.sample_sources,
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
            {"proposed_concepts": len(concepts), "awaiting_reading": True,
             "hidden_ungrounded": proposal.hidden,
             "evidence_shown": [row["passage_id"] for row in proposal.evidence],
             "keyphrases_shown": proposal.keyphrases}, indent=2
        ).encode()
        artifacts["answer.md"] = b"Choose one of the agent's concept readings to continue."
        artifacts["trace.json"] = json.dumps(
            [row.model_dump(by_alias=True) for row in trace], indent=2
        ).encode()
        return {"run": run, "artifacts": artifacts}


__all__ = ["DEFAULT_CLASSES", "ConceptProposalStep"]
