"""Model capacity: run slots, and public-demo reservations taken before a run starts.

``MODEL_SLOT`` bounds concurrent research runs (``HARNESS_MAX_CONCURRENT_RUNS``).
Public runs also hold ``PUBLIC_MODEL_SLOT``; at least one total slot is always
kept for private work.
"""

from __future__ import annotations

import ipaddress
import secrets
import threading
from time import monotonic

from fastapi import HTTPException, Request
from harness_config import settings

_config = settings()
HEAVY_MODEL_SLOT = threading.BoundedSemaphore(_config.max_concurrent_heavy)
RUN_CAPACITY = _config.max_concurrent_runs
MODEL_SLOT = threading.BoundedSemaphore(RUN_CAPACITY)
# Reserve at least one total slot for private work. If total capacity is one,
# public work stays disabled instead of taking the only researcher slot.
PUBLIC_RUN_CAPACITY = max(0, min(_config.max_concurrent_public_runs, RUN_CAPACITY - 1))
PUBLIC_MODEL_SLOT = threading.BoundedSemaphore(PUBLIC_RUN_CAPACITY)
PUBLIC_RESERVATION_TTL_S = 15.0
PUBLIC_RESERVATIONS: dict[str, tuple[float, tuple[threading.BoundedSemaphore, ...]]] = {}
PUBLIC_RESERVATIONS_LOCK = threading.Lock()


def run_slots(origin: str) -> tuple[threading.BoundedSemaphore, ...]:
    """The slots a run of this origin must hold while it researches."""
    return (PUBLIC_MODEL_SLOT, MODEL_SLOT) if origin == "public" else (MODEL_SLOT,)


def require_loopback(request: Request) -> None:
    peer = request.client.host if request.client else ""
    try:
        if not ipaddress.ip_address(peer).is_loopback:
            raise HTTPException(403, "forbidden")
    except ValueError as exc:
        raise HTTPException(403, "forbidden") from exc


def release_expired_public_reservations() -> None:
    now = monotonic()
    with PUBLIC_RESERVATIONS_LOCK:
        expired = [
            token
            for token, (created, _) in PUBLIC_RESERVATIONS.items()
            if now - created > PUBLIC_RESERVATION_TTL_S
        ]
        for token in expired:
            _, slots = PUBLIC_RESERVATIONS.pop(token)
            for slot in slots:
                slot.release()


def consume_public_reservation(token: str | None):
    release_expired_public_reservations()
    if not token:
        return None
    with PUBLIC_RESERVATIONS_LOCK:
        reservation = PUBLIC_RESERVATIONS.pop(token, None)
    return reservation[1] if reservation else None


def public_capacity_available() -> bool:
    release_expired_public_reservations()
    public_ok = PUBLIC_MODEL_SLOT.acquire(blocking=False)
    model_ok = MODEL_SLOT.acquire(blocking=False) if public_ok else False
    if model_ok:
        MODEL_SLOT.release()
    if public_ok:
        PUBLIC_MODEL_SLOT.release()
    return public_ok and model_ok


def reserve_public_capacity() -> str:
    """Atomically reserve public and total slots; raises 503 when busy."""
    release_expired_public_reservations()
    if not PUBLIC_MODEL_SLOT.acquire(blocking=False):
        raise HTTPException(503, "service busy")
    if not MODEL_SLOT.acquire(blocking=False):
        PUBLIC_MODEL_SLOT.release()
        raise HTTPException(503, "service busy")
    token = secrets.token_urlsafe(24)
    with PUBLIC_RESERVATIONS_LOCK:
        PUBLIC_RESERVATIONS[token] = (monotonic(), (PUBLIC_MODEL_SLOT, MODEL_SLOT))
    return token
