"""Independent verification: a fresh-context check of the agent's cited evidence."""

from __future__ import annotations

import json
from typing import Any

from harness_extraction import model_purpose, using_prompt
from harness_prompts import render
from harness_schema.models import Check, Run

# The id that names a row, in order; ``paper_id`` names a row only when it has no other id
# (a paper's metadata), otherwise it is the paper the row belongs to.
HANDLE_KEYS = ("triple_id", "passage_id", "fact_id", "reference_id", "source_id", "claim_id", "paper_id")
TEXT_KEYS = ("text", "evidence", "snippet", "title")
EXCERPT_CHARS = 600
# How directly a result shows an id (lower is better): the row it names, another id of a row
# (a triple's source passage, a passage's paper), a member of a row's ``*_ids`` list.
NAMED, IN_ROW, LISTED = range(3)


def _row_text(row: dict[str, Any]) -> str:
    """What a result row says: a triple's statement with its passage, a passage's text, or its fields."""
    passage = next((str(row[key]) for key in TEXT_KEYS if row.get(key)), "")
    if {"subject", "predicate", "object"} <= row.keys():
        statement = f"{row['subject']} {row['predicate']} {row['object']}"
        return f"{statement} — passage: {passage}" if passage and passage != statement else statement
    if passage:
        return passage
    fields = [f"{key}: {value}" for key, value in row.items()
              if isinstance(value, (str, int, float)) and not isinstance(value, bool) and str(value).strip()]
    fields += [f"{key}: {len(value)} ids" for key, value in row.items()
               if key.endswith("_ids") and isinstance(value, list)]
    return " · ".join(fields) or json.dumps(row, ensure_ascii=False, default=str)


def excerpt_index(calls: list[dict], wanted: set[str]) -> dict[str, str]:
    """The text behind each wanted id, from this session's tool results.

    Corpus and document tool results are searched for every way a row can show an id: the id it is
    named by (``HANDLE_KEYS``, ``triple_id`` included), its other ``*_id`` fields and the entries of
    its ``*_ids`` lists (mapped to that row's text). A paper reader's record contributes the passages
    it read (``passages``), not its findings. The most direct mention wins, then the longest text.
    """
    found: dict[str, tuple[int, str]] = {}

    def offer(handle: str, rank: int, text: str) -> None:
        if handle not in wanted:
            return
        text = text[:EXCERPT_CHARS]
        earlier = found.get(handle)
        if earlier is None or rank < earlier[0] or (rank == earlier[0] and len(text) > len(earlier[1])):
            found[handle] = (rank, text)

    def walk(value: object) -> None:
        if isinstance(value, dict):
            handle_key = next((key for key in HANDLE_KEYS if isinstance(value.get(key), (str, int))
                               and str(value[key]).strip()), None)
            if handle_key is not None:
                text = _row_text(value)
                offer(str(value[handle_key]), NAMED, text)
                for key, item in value.items():
                    if key != handle_key and key.endswith("_id") and isinstance(item, (str, int)):
                        offer(str(item), IN_ROW, f"[{handle_key} {value[handle_key]}] {text}")
            lists = {key: item for key, item in value.items() if key.endswith("_ids") and isinstance(item, list)}
            if lists:
                text = _row_text(value)
                for item in lists.values():
                    for entry in item:
                        if isinstance(entry, (str, int)):
                            offer(str(entry), LISTED, f"[listed in] {text}")
            for item in value.values():
                if isinstance(item, (dict, list)):
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    for call in calls:
        result = call.get("result")
        if call.get("tool") == "paper_reader":
            walk(result.get("passages") if isinstance(result, dict) else None)
        else:
            walk(result)
    return {handle: text for handle, (_rank, text) in found.items()}


def independent_verification(
    verifier: Any, question: str, decision: dict, verified: list, evidence_ids: list,
    corpus_calls: list[dict], base_run: Run,
) -> tuple[dict | None, Check]:
    """One structured call, in a fresh context, judging whether the cited evidence supports the decision.

    The verifier never sees the agent's conversation: only the question, the decision (its
    reasoning summary, which the caller extends with the finding statements, and its ids) and the
    text of the evidence items it cites, looked up in this session's tool results (``corpus_calls``:
    corpus or document tool records and paper-reader records, ``excerpt_index``) and in the
    preliminary claims' evidence. It cannot add evidence, only mark cited items supported or not,
    and it never fails a run. A cited id found in none of them is reported unsupported unjudged.
    """
    complete = getattr(verifier, "complete_json", None)
    if complete is None:
        return None, Check(name="independent verification", status="skipped", detail="no verifier model configured")
    wanted = {str(item) for item in evidence_ids}
    excerpts = excerpt_index(corpus_calls, wanted)
    for reading in base_run.readings:
        for claim in reading.claims:
            for source, text in claim.evidence.items():
                if source in wanted and source not in excerpts and text:
                    excerpts[source] = str(text)[:EXCERPT_CHARS]
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
