"""Independent verification: a fresh-context check of the agent's cited evidence."""

from __future__ import annotations

import json
from typing import Any

from harness_extraction import model_purpose, using_prompt
from harness_prompts import render
from harness_schema.models import Check, Run


def independent_verification(
    verifier: Any, question: str, decision: dict, verified: list, evidence_ids: list,
    corpus_calls: list[dict], base_run: Run,
) -> tuple[dict | None, Check]:
    """One structured call, in a fresh context, judging whether the cited evidence supports the decision.

    The verifier never sees the agent's conversation: only the question, the
    decision and the text of the evidence items it cites (looked up in this
    session's tool results and preliminary claims). It cannot add evidence,
    only mark cited items supported or not, and it never fails a run.
    """
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
    prompt = render("agent/independent_verifier", request, ensure_ascii=False)
    try:
        with model_purpose("independent verification"), using_prompt(prompt.ref):
            payload = complete(prompt.system, prompt.user)
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
              "supported_ids": supported, "unsupported_ids": unsupported, "notes": notes,
              "prompt": prompt.ref}
    if not cited:
        status, detail = "skipped", "the decision cites nothing this session returned"
    elif unsupported:
        status = "warn"
        detail = (f"{len(supported)} of {len(cited)} cited items judged to support the decision; "
                  f"not supported: {', '.join(unsupported[:5])}" + (f" — {notes}" if notes else ""))
    else:
        status, detail = "pass", f"all {len(cited)} cited items judged to support the decision" + (f" — {notes}" if notes else "")
    return result, Check(name="independent verification", status=status, detail=detail)


__all__ = ["independent_verification"]
