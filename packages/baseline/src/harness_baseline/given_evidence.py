"""The evidence document D as the input of a Pipeline run (docs/RUN_INPUT.md). A mixin of ``Baseline20``.

Instead of retrieval and extraction: D's triples (with their passages) go to the evidence review, which
picks the ones that are supported and answer the question, in batches of ``HARNESS_EVIDENCE_DOC_BATCH``.
Every path picks, Fast included: otherwise every triple of D would be its answer. Then the usual answer
composition (with verification on Adaptive and Agentic) and the usual answer graph.
"""

from __future__ import annotations

from typing import Any

from harness_config import settings
from harness_extraction.schemas import ExtractedTriple, SchemaInference

from .evidence_doc import EvidenceDoc
from .state import BaselineState, NoResults


class GivenEvidence:
    evidence_doc: EvidenceDoc | None

    def _doc_relations(self, triples: list[ExtractedTriple]) -> tuple[list[ExtractedTriple], int]:
        """D's relation names, registered as run-local properties where the schema lacks them (as for any
        invented property); a name that is not a valid property name is dropped and counted."""
        kept, invalid = [], 0
        for triple in triples:
            name = triple.predicate
            if name not in self.extraction.property_names:
                try:
                    self.extraction.register_local_relation(
                        name, f"Relation of the evidence document: {name}.", domain_name="Paper",
                        range_name="Paper" if triple.object in self.corpus.by_paper else "xsd:string")
                except ValueError:
                    invalid += 1
                    continue
            kept.append(triple)
        return kept, invalid

    def _doc_state(self, question: str) -> BaselineState:
        doc = self.evidence_doc
        assert doc is not None
        self.stage = "evidence document"
        triples, invalid = self._doc_relations(doc.extracted())
        passages = [self.corpus.by_passage[source] for source in doc.source_ids() if source in self.corpus.by_passage]
        where = f"{doc.question_id}" + (f", interpretation {doc.reading}" if doc.reading else "") + (
            f", {doc.chosen} chosen by the agent" if doc.chosen is not None else "")
        self._emit("Reading the evidence document", f"{len(triples)} triples from {len(passages)} passages ({where})")
        self._check("retrieval", "skipped", "the input is the evidence document D (run input: evidence_doc)")
        notes = [f"{doc.left_out} without a corpus passage left out" if doc.left_out else "",
                 f"{invalid} with an unusable relation name left out" if invalid else ""]
        self._check("extraction", "skipped", "; ".join(["D's triples are the input", *filter(None, notes)]))
        if not triples:
            raise NoResults("the evidence document has no usable triple for this question")
        base: BaselineState = {"question": question, "concept": self.concept,
                               "agent_findings": list(self.agent_findings)}
        self.stage = "evidence review"
        kept, review = self.extraction.refine_triples(
            self._research_question(base), triples, passages, batch_size=settings().evidence_doc_batch,
            reviewer=self.refinement_model)
        labelled: list[dict[str, Any]] = review.pop("labelled", [])
        self._emit("Picking the triples that answer", f"{len(kept)} of {review['unique']} kept")
        self._check("evidence review", "pass" if kept else "fail",
                    f"{len(kept)} of {review['unique']} triples of D picked ({review['model_calls']} calls)")
        if not kept:
            raise NoResults("the evidence review picked no triple of the evidence document")
        inferred = SchemaInference(
            label="Evidence document", gloss=f"Triples of the evidence document of {doc.question_id}.",
            classes=["Paper"], properties=list(dict.fromkeys(item.predicate for item in kept)))
        state: BaselineState = {
            **base, "plan": None, "paper_ids": [], "computed": [], "passages": passages,
            "inferred": inferred, "extracted": kept, "evidence_graph": labelled,
            "retrieval": {"stop_reason": "evidence_doc", "queries": [], "evidence_doc": doc.summary()},
            "extraction_batches": [{"stage": "evidence_review", "input": "evidence_doc",
                                    "triples": len(triples), **review}],
        }
        state.update(self._query_answer(state))
        return state


__all__ = ["GivenEvidence"]
