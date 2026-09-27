"""Model status for the UI: load, queue and availability of each model, plus the harness's own run queue.

The numbers come from the thesis-llm-setup status API (``LLM_STATUS_URL``), which polls every vLLM
backend every 5 s; this endpoint only relays them (the browser cannot reach that port through the
tunnel) and adds which harness roles use each model and how many runs hold or wait for a slot.
"""

from __future__ import annotations

from typing import Any

import app_state
import httpx
from capacity import MODEL_SLOT, RUN_CAPACITY
from fastapi import APIRouter
from harness_config import settings

router = APIRouter()
FIELDS = ("state", "running", "waiting", "kvCache", "queueP95Seconds", "latencyP95Seconds", "preemptions",
          "lastHealthyAt")
# Other spellings of the counts (older status APIs, Prometheus-style names); the first one present wins.
SPELLINGS = {"running": ("running", "requestsRunning", "numRequestsRunning", "num_requests_running"),
             "waiting": ("waiting", "queued", "requestsWaiting", "numRequestsWaiting", "num_requests_waiting")}


def _value(entry: dict[str, Any], key: str) -> Any:
    return next((entry[name] for name in SPELLINGS.get(key, (key,)) if entry.get(name) is not None), None)


_PROVIDERS = ("openai/", "hosted_vllm/", "vllm/", "litellm_proxy/")


def _alias(name: str) -> str:
    """"openai/heavy-model" and "heavy-model" are one model (a LiteLLM provider prefix is dropped;
    "Qwen/Qwen3-8B" keeps its organisation)."""
    return next((name[len(prefix):] for prefix in _PROVIDERS if name.startswith(prefix)), name)


def model_roles() -> dict[str, list[str]]:
    """Which harness roles each model alias serves (from the HARNESS_MODEL_* settings)."""
    config = settings()
    roles: dict[str, list[str]] = {}
    for role, alias in (("agent", config.model_agent), ("schema and answers", config.model_high),
                        ("extraction", config.model_edc or config.model_small),
                        ("review and verification", config.model_judge), ("embeddings", config.model_embedding)):
        if alias:
            roles.setdefault(alias, []).append(role)
    return roles


def run_queue() -> dict[str, int]:
    runtime = app_state.runtime
    with runtime.lock:
        queued = len(runtime.queued)
    # Slots in use, not runs in the process: proposal and finalisation hold no research slot.
    active = max(0, RUN_CAPACITY - MODEL_SLOT._value)  # BoundedSemaphore has no public count
    return {"active": active, "queued": queued, "capacity": RUN_CAPACITY}


@router.get("/models/status")
async def models_status() -> dict[str, Any]:
    """``reachable`` false (with ``detail``) when the status API cannot be read; models then list the
    configured aliases with state ``unknown``."""
    url = settings().llm_status_url.strip()
    roles = model_roles()
    document: dict[str, Any] = {}
    detail = None
    if not url:
        detail = "LLM_STATUS_URL is not set."
    else:
        try:
            async with httpx.AsyncClient(timeout=2.5) as client:
                response = await client.get(url)
                response.raise_for_status()
                body = response.json()
                document = body if isinstance(body, dict) else {}
        except (httpx.HTTPError, ValueError) as exc:
            detail = f"The status API at {url} is unreachable ({type(exc).__name__})."
    raw = document.get("models") if isinstance(document.get("models"), dict) else {}
    reported = {_alias(str(name)): entry for name, entry in raw.items() if isinstance(entry, dict)}
    merged: dict[str, list[str]] = {}
    for alias, names in roles.items():
        merged.setdefault(_alias(alias), []).extend(name for name in names if name not in merged.get(_alias(alias), []))
    roles = merged
    aliases = list(dict.fromkeys([*roles, *reported]))
    models = []
    for alias in aliases:
        entry = reported.get(alias, {})
        models.append({"alias": alias, "roles": roles.get(alias, []),
                       **{key: _value(entry, key) for key in FIELDS},
                       "state": entry.get("state") or "unknown"})
    return {"reachable": detail is None, "detail": detail, "overall": document.get("overall", "unknown"),
            "updatedAt": document.get("updatedAt"), "models": models, "runs": run_queue()}
