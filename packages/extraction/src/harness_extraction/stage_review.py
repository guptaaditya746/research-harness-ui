"""Evidence review: keep only supported triples relevant to the question, with the reason for each drop.

A mixin of ``ExtractionEngine``.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from harness_config import settings
from harness_prompts import render

from .call_control import parallel_map
from .events import labelled
from .model_client import ModelClient, complete
from .schemas import ExtractedTriple


def _drop_reason(value: object) -> str:
    """Evidence review's reason for dropping a triple, as one of the prompt's words."""
    text = str(value or "").strip().casefold()
    return next((word for word in ("unsupported", "irrelevant", "malformed") if word in text),
                "no reason given" if not text else "other")



class ReviewStage:
    @labelled("evidence review")
    def refine_triples(
        self,
        question: str,
        triples: list[ExtractedTriple],
        passages: list[dict[str, Any]],
        batch_size: int = 24,
        reviewer: ModelClient | None = None,
    ) -> tuple[list[ExtractedTriple], dict[str, Any]]:
        """Second model pass: retain only atomic claims relevant to the question.

        The reviewer receives numbered existing candidates and may only return
        their IDs. It cannot add a fact or change provenance. Batches cover the
        entire candidate set, so adaptive retrieval has no output cap here.
        """
        passage_text = {str(row["passage_id"]): str(row["text"]) for row in passages}
        deduped: dict[tuple[str, str, str], ExtractedTriple] = {}
        for item in triples:
            if not all((item.subject.strip(), item.predicate.strip(), item.object.strip(), item.source_id.strip())):
                continue
            key = (item.subject.casefold().strip(), item.predicate, item.object.casefold().strip())
            deduped.setdefault(key, item)
        candidates = list(deduped.values())
        batches = [candidates[start : start + batch_size] for start in range(0, len(candidates), batch_size)]

        def review(batch: list[ExtractedTriple]) -> tuple[list[ExtractedTriple], list[dict[str, str]]]:
            payload = {
                "question": question,
                "candidates": [
                    {
                        "id": i,
                        **item.model_dump(),
                        "source_text": passage_text.get(item.source_id, "")[:1500],
                    }
                    for i, item in enumerate(batch)
                ],
            }
            decision = complete(reviewer or self.model,
                                render("extraction/evidence_review", payload, ensure_ascii=False))
            selected = decision.get("keep_ids", []) if isinstance(decision, dict) else []
            if not isinstance(selected, list):
                raise TypeError("triple reviewer must return keep_ids as a list")
            indices = {value for value in selected if type(value) is int and 0 <= value < len(batch)}
            reasons = decision.get("drop_reasons") if isinstance(decision, dict) else None
            reasons = reasons if isinstance(reasons, dict) else {}
            dropped = [{"triple": f"{item.subject} —{item.predicate}→ {item.object}", "source": item.source_id,
                        "why": _drop_reason(reasons.get(str(i))), "index": id(item)}
                       for i, item in enumerate(batch) if i not in indices]
            return [item for i, item in enumerate(batch) if i in indices], dropped

        # Review batches run in parallel (HARNESS_EXTRACTION_CONCURRENCY); kept triples keep their order.
        results = parallel_map(review, batches, settings().extraction_concurrency)
        kept = [item for part, _ in results for item in part]
        dropped = [item for _, part in results for item in part]
        why_dropped = {row.pop("index"): row["why"] for row in dropped}
        report: dict[str, Any] = {"candidates": len(triples), "unique": len(candidates), "kept": len(kept),
                                  "model_calls": len(batches)}
        # Every candidate with the review's verdict: the run's evidence graph keeps what the answer drops.
        report["labelled"] = [
            {**item.model_dump(), "relevant": id(item) not in why_dropped,
             **({"why": why_dropped[id(item)]} if id(item) in why_dropped else {})}
            for item in candidates
        ]
        if dropped:
            # Why facts are lost at this stage (the largest loss in the funnel): counts per reason and examples.
            report["dropped"] = dict(Counter(item["why"] for item in dropped).most_common())
            report["dropped_examples"] = dropped[:8]
        return kept, report
