"""How busy the high (Heavy) model is, from the thesis-llm-setup status API."""

from __future__ import annotations

import httpx
from harness_config import settings


def heavy_queue_depth() -> int | None:
    """Requests waiting on the high model, or None when load is unknown.

    Reads the thesis-llm-setup status API (``LLM_STATUS_URL``), the one
    supported source of model load; a saturated model counts as at least
    the queue threshold. Falls back to vLLM's own metrics at
    ``VLLM_HEAVY_API_BASE`` only when the status API is unreachable.
    """
    config = settings()
    alias = config.model_high
    threshold = config.heavy_queue_threshold
    status_url = config.llm_status_url.strip()
    if status_url:
        try:
            response = httpx.get(status_url, timeout=3)
            response.raise_for_status()
            entry = (response.json().get("models") or {}).get(alias)
            if isinstance(entry, dict):
                if entry.get("state") == "unavailable":
                    return None
                waiting = int(entry.get("waiting") or 0)
                return max(waiting, threshold) if entry.get("state") == "saturated" else waiting
        except (httpx.HTTPError, ValueError, TypeError, AttributeError):
            pass
    base = config.vllm_heavy_api_base.rstrip("/")
    if not base:
        return None
    endpoint = base.removesuffix("/v1") + "/metrics"
    try:
        response = httpx.get(
            endpoint, headers={"Authorization": f"Bearer {config.api_key}"}, timeout=3,
        )
        response.raise_for_status()
        for line in response.text.splitlines():
            if line.startswith("vllm:num_requests_waiting{"):
                return int(float(line.rsplit(" ", 1)[-1]))
    except (httpx.HTTPError, ValueError):
        return None
    return None


__all__ = ["heavy_queue_depth"]
