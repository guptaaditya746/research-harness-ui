"""Build provenance-bearing research claims from verified graph edges.

The original edge is retained as a field, so missing scientific context is
visible as a partial claim rather than filled with a guessed value.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from typing import Any

from harness_config import settings
from harness_prompts import render
from harness_schema.models import ClaimField, ClaimRelation, ClaimSubgraph, EvidenceAnchor, Triple

from .call_control import parallel_map
from .events import labelled
from .model_client import complete

PLACEHOLDERS = {"", "null", "none", "n/a", "unknown", "undefined"}
NUMERIC = re.compile(r"(?<!\w)(\d+(?:\.\d+)?)\s*(%|percent(?:age points?)?)", re.IGNORECASE)
METRICS = re.compile(r"\b(success rate|accuracy|error rate|f1|precision|recall|completion rate)\b", re.IGNORECASE)
CHANGE = re.compile(r"\b(relative improvement|improvement|increase|decrease|reduction|gain)\b", re.IGNORECASE)


def valid_edge(triple: Triple) -> bool:
    return all(str(value).strip().casefold() not in PLACEHOLDERS for value in
               (triple.s, triple.p, triple.o, triple.source, triple.evidence))


def _anchor(source: str, paper: str, evidence: str) -> EvidenceAnchor:
    kind = "dblp_fact" if source.startswith("dblp:") or source.endswith("-metadata") or evidence.startswith("Corpus fact ") else "passage"
    return EvidenceAnchor(
        id=f"evidence:{source}", kind=kind, paperId=paper, sourceId=source,
        locator={"fact_id": evidence.removeprefix("Corpus fact ").rstrip(".") if kind == "dblp_fact" else source}
        if kind == "dblp_fact" else {"passage_id": source},
        text=evidence,
    )


def _field(value: str, source: str) -> ClaimField:
    return ClaimField(value=value.strip(), sourceIds=[source], evidenceIds=[f"evidence:{source}"])


def _kind(triple: Triple) -> str:
    predicate = triple.p.casefold()
    if predicate in {"title", "author", "authoredby", "year", "venue", "cites", "citestitle"}:
        return "MetadataClaim"
    if "limitation" in predicate or "limitation" in triple.o.casefold():
        return "LimitationClaim"
    if "compar" in predicate:
        return "ComparisonClaim"
    if "metric" in predicate or NUMERIC.search(triple.o):
        return "ResultClaim"
    return "MethodClaim"


def build_claims(
    triples: list[Triple],
    paper_for_source: Callable[[str, str], str],
) -> tuple[list[ClaimSubgraph], list[ClaimRelation]]:
    """Create one attributed claim per distinct assertion and context.

    Exact repeated assertions within a paper merge provenance; assertions from
    different papers remain separate even if their wording is identical.
    """
    claims: dict[str, ClaimSubgraph] = {}
    for triple in triples:
        if not valid_edge(triple):
            continue
        paper = paper_for_source(triple.source, triple.s)
        kind = _kind(triple)
        fields = {
            "subject": _field(triple.s, triple.source),
            "relation": _field(triple.p, triple.source),
            "object": _field(triple.o, triple.source),
        }
        number = NUMERIC.search(triple.o)
        if kind == "ResultClaim" and number:
            fields["value"] = _field(number.group(1), triple.source)
            fields["unit"] = _field("percent" if number.group(2) == "%" else number.group(2), triple.source)
            metric = METRICS.search(f"{triple.o} {triple.evidence}")
            change = CHANGE.search(f"{triple.o} {triple.evidence}")
            if metric:
                fields["metric"] = _field(metric.group(), triple.source)
            if change:
                fields["changeType"] = _field(change.group(), triple.source)
        identity = f"{kind}\x1f{paper}\x1f{triple.s.casefold()}\x1f{triple.p.casefold()}\x1f{triple.o.casefold()}"
        claim_id = "claim-" + hashlib.sha256(identity.encode()).hexdigest()[:16]
        previous = claims.get(claim_id)
        if previous:
            sources = sorted(set(previous.source_ids) | {triple.source})
            merged_fields = {
                key: field.model_copy(update={"source_ids": sources})
                for key, field in previous.fields.items()
            }
            claims[claim_id] = previous.model_copy(update={
                "source_ids": sources, "evidence": {**previous.evidence, triple.source: triple.evidence},
                "fields": merged_fields,
                "evidence_anchors": previous.evidence_anchors + [
                    _anchor(triple.source, paper, triple.evidence)
                ] if not any(anchor.id == f"evidence:{triple.source}" for anchor in previous.evidence_anchors)
                else previous.evidence_anchors,
            })
        else:
            claims[claim_id] = ClaimSubgraph(
                id=claim_id, kind=kind, paperId=paper, fields=fields,
                sourceIds=[triple.source], evidence={triple.source: triple.evidence},
                evidenceAnchors=[_anchor(triple.source, paper, triple.evidence)],
                status="partial" if kind == "ResultClaim" and
                not {"metric", "value", "unit"} <= set(fields) else "accepted",
            )
    rows = list(claims.values())
    return rows, classify_claim_relations(rows)


def classify_claim_relations(rows: list[ClaimSubgraph]) -> list[ClaimRelation]:
    """Classify result disagreements after all available context is compared."""
    relations: list[ClaimRelation] = []
    for index, left in enumerate(rows):
        if left.kind != "ResultClaim":
            continue
        for right in rows[index + 1:]:
            if right.kind != "ResultClaim" or left.paper_id != right.paper_id:
                continue
            if left.fields["subject"].value != right.fields["subject"].value:
                continue
            if left.fields["object"].value != right.fields["object"].value:
                left_context = {name: field.value.casefold() for name, field in left.fields.items()
                                if name in {"metric", "benchmark", "setting", "unit", "comparator"}}
                right_context = {name: field.value.casefold() for name, field in right.fields.items()
                                 if name in {"metric", "benchmark", "setting", "unit", "comparator"}}
                comparable_context = {"metric", "benchmark", "setting", "unit"}
                relations.append(ClaimRelation(
                    sourceClaim=left.id, targetClaim=right.id,
                    relation=("confirmedConflict" if comparable_context <= set(left_context) and
                              left_context == right_context else "possibleConflict"),
                    reason=("Different values under matching material context."
                            if comparable_context <= set(left_context) and left_context == right_context
                            else "Different reported results; evaluation contexts require comparison."),
                ))
    return relations


def derive_query_claim(question: str, claims: list[ClaimSubgraph]) -> ClaimSubgraph | None:
    """Add an explicit harness inference above source-specific claims.

    This intentionally never merges reported claims.  It records the aggregation
    or synthesis used by a question so its prose can distinguish a corpus result
    from something asserted by any one paper.
    """
    if not claims:
        return None
    folded = question.casefold()
    kind = None
    if re.search(r"\b(best|most|highest|rank|ranking|more than)\b", folded):
        kind = "RankingClaim"
        relation, object_ = "aggregation", "corpus ranking/threshold result"
    elif re.search(r"\b(across|shared|common|these papers|synthes)\b", folded):
        kind = "SynthesisClaim"
        relation, object_ = "synthesis", "cross-paper corpus synthesis"
    if kind is None:
        return None
    source_ids = sorted({source for claim in claims for source in claim.source_ids})
    evidence = {source: claim.evidence[source] for claim in claims for source in claim.source_ids}
    fields = {
        "subject": _field("indexed corpus", source_ids[0]),
        "relation": _field(relation, source_ids[0]),
        "object": _field(object_, source_ids[0]),
        "criterion": _field(question.strip(), source_ids[0]),
        "corpusScope": _field("indexed corpus snapshot", source_ids[0]),
    }
    identity = f"{kind}\x1f{question.casefold()}\x1f{'|'.join(claim.id for claim in claims)}"
    return ClaimSubgraph(
        id="claim-" + hashlib.sha256(identity.encode()).hexdigest()[:16], kind=kind,
        paperId="https://harness.local/resource/corpus", fields=fields,
        sourceIds=source_ids, evidence=evidence,
        evidenceAnchors=[anchor for claim in claims for anchor in claim.evidence_anchors],
        supportedBy=[claim.id for claim in claims] if kind == "SynthesisClaim" else [],
        status="accepted",
    )


@labelled("claim context enrichment")
def enrich_result_claims(
    claims: list[ClaimSubgraph], passages: list[dict[str, Any]], model: Any,
) -> list[ClaimSubgraph]:
    """Extract context fields only when the model supplies an exact evidence quote."""
    candidates = [claim for claim in claims if claim.kind in {"ResultClaim", "ComparisonClaim"}]
    if not candidates:
        return claims
    by_source = {str(row["passage_id"]): str(row["text"]) for row in passages}
    allowed = {"method", "metric", "benchmark", "setting", "comparator", "changeType", "value", "unit"}
    enriched = {claim.id: claim for claim in claims}

    def ask(batch: list[ClaimSubgraph]) -> Any:
        return complete(model, render("extraction/claim_enrichment", {"claims": [{
            "id": claim.id, "paper_id": claim.paper_id,
            "fields": {name: field.value for name, field in claim.fields.items()},
            "sources": {source: by_source.get(source, excerpt)[:2500]
                        for source, excerpt in claim.evidence.items()},
        } for claim in batch]}, ensure_ascii=False))

    # Batches of six claims, sent HARNESS_EXTRACTION_CONCURRENCY at a time (many claims: the evidence
    # document as input can give hundreds); each batch changes only its own claims.
    batches = [candidates[start:start + 6] for start in range(0, len(candidates), 6)]
    for response in parallel_map(ask, batches, settings().extraction_concurrency):
        if not isinstance(response, dict):
            continue
        for item in response.get("enrichments", []):
            if not isinstance(item, dict) or item.get("claim_id") not in enriched:
                continue
            claim = enriched[item["claim_id"]]
            fields = dict(claim.fields)
            proposed = item.get("fields")
            if not isinstance(proposed, dict):
                continue
            for name, value in proposed.items():
                if name not in allowed or name in fields or not isinstance(value, dict):
                    continue
                source = str(value.get("source_id") or "")
                quote = str(value.get("quote") or "").strip()
                content = by_source.get(source, claim.evidence.get(source, ""))
                field_value = str(value.get("value") or "").strip()
                if source not in claim.source_ids or not quote or quote.casefold() not in content.casefold():
                    continue
                if field_value.casefold() not in quote.casefold() or field_value.casefold() in PLACEHOLDERS:
                    continue
                fields[name] = ClaimField(value=field_value, sourceIds=[source], quote=quote,
                                          evidenceIds=[f"evidence:{source}"])
            status = "accepted" if {"metric", "value", "unit"} <= set(fields) else "partial"
            enriched[claim.id] = claim.model_copy(update={"fields": fields, "status": status})
    return [enriched[claim.id] for claim in claims]


__all__ = ["build_claims", "classify_claim_relations", "derive_query_claim", "enrich_result_claims", "valid_edge"]
