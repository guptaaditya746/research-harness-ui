"""What the research agent is offered and what its final decision must look like.

The curated tool list, the ``record_finding`` scratchpad tool, and the deterministic checks of
findings and of the final decision (sent back to the agent with their problems).
"""

from __future__ import annotations

import json
from typing import Any

DECISION_KEYS = ("verified_claim_ids", "additional_evidence_ids", "reasoning_summary")
# The tools offered to the agent: one tool per job. Left out are
# duplicates (rank, rank_evidence, candidates_for, query2doc = search_evidence; facts =
# structured_facts; read_passage, expand = read_evidence) and rarely useful specialists.
CURATED_TOOLS = (
    "search_evidence", "read_evidence", "structured_facts", "aggregate_facts", "metadata_table",
    "citations", "evidence_for_claim", "search", "grep", "paper_outline", "abstracts", "list_papers",
    "table_search", "results",
)
FINDING_TOOL = "record_finding"
# A tool the client answers itself (no MCP round trip): the agent's scratchpad.
FINDING_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": FINDING_TOOL,
        "description": (
            "Save one finding to your research notes: a short statement of what the evidence shows and the "
            "evidence ids (returned by tools in this session) that support it. Notes are kept when older tool "
            "results are removed from the conversation, and they are handed to the answer pipeline. Record a "
            "finding as soon as you have read the evidence for it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "statement": {"type": "string", "description": "One or two sentences, specific (names, numbers)."},
                "evidence_ids": {"type": "array", "items": {"type": "string"},
                                 "description": "Passage, fact or reference ids a tool returned."},
            },
            "required": ["statement", "evidence_ids"],
        },
    },
}


def id_list(value: object) -> list[str]:
    """Ids from a model-written field: a list, or one id given as a string or number."""
    if isinstance(value, str | int) and not isinstance(value, bool):
        value = [value]
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value
            if isinstance(item, str | int) and not isinstance(item, bool) and str(item).strip()]


def validate_findings(raw: object, observed: set[str]) -> tuple[list[dict[str, Any]], list[str]]:
    """Findings with at least one observed id (unobserved ids removed), and the problems found."""
    findings: list[dict[str, Any]] = []
    problems: list[str] = []
    for index, item in enumerate(raw if isinstance(raw, list) else []):
        if not isinstance(item, dict) or not str(item.get("statement") or "").strip():
            problems.append(f"finding {index + 1} needs a statement")
            continue
        ids = list(dict.fromkeys(id_list(item.get("evidence_ids"))))
        kept = [value for value in ids if value in observed]
        if not kept:
            problems.append(f"finding {index + 1} cites no id returned by a tool in this session")
            continue
        findings.append({"statement": str(item["statement"]).strip()[:400], "evidence_ids": kept[:8]})
    return findings, problems


def decision_problems(
    final: str, *, observed: set[str], claim_ids: set[str], corpus_calls: int, needs_tools: bool,
) -> list[str]:
    """Why the agent's final decision cannot be accepted as-is (empty when it can).

    Checked deterministically before the decision leaves the tool loop, so the
    agent gets one chance to repair it instead of the run degrading.
    """
    start, end = final.find("{"), final.rfind("}")
    try:
        decision = json.loads(final[start:end + 1]) if start >= 0 else None
    except json.JSONDecodeError:
        decision = None
    if not isinstance(decision, dict):
        return ["the final message must be one JSON object with keys " + ", ".join(DECISION_KEYS)]
    problems: list[str] = []
    for key in ("verified_claim_ids", "additional_evidence_ids"):
        if not isinstance(decision.get(key), list):
            problems.append(f"{key} must be a list (use [] when empty)")
    evidence = id_list(decision.get("additional_evidence_ids")) if isinstance(decision.get("additional_evidence_ids"), list) else []
    unobserved = [item for item in evidence if item not in observed]
    if unobserved:
        problems.append(
            "these additional_evidence_ids were never returned by any tool in this session and must be "
            "removed or replaced by ids you actually read: " + ", ".join(unobserved[:8])
        )
    verified = [str(item) for item in decision.get("verified_claim_ids") or [] if isinstance(decision.get("verified_claim_ids"), list)]
    unknown = [item for item in verified if item not in claim_ids]
    if unknown:
        problems.append(
            "these verified_claim_ids are not preliminary claim ids (allowed: "
            + ", ".join(sorted(claim_ids)[:8]) + "): " + ", ".join(unknown[:8])
        )
    if needs_tools and corpus_calls == 0:
        problems.append("no corpus tool was called; gather evidence with at least one tool call before deciding")
    if "findings" in decision:
        if not isinstance(decision["findings"], list):
            problems.append("findings must be a list of {statement, evidence_ids} (use [] when empty)")
        else:
            problems.extend(validate_findings(decision["findings"], observed)[1])
    return problems
