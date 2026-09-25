"""Loopback-only seam for the public gateway: capacity checks and reservations."""

from __future__ import annotations

from capacity import public_capacity_available, require_loopback
from capacity import reserve_public_capacity as reserve
from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/internal/public/capacity")
def public_capacity(request: Request) -> dict[str, bool]:
    """Reports only whether public work is accepted; no private queue or model metrics."""
    require_loopback(request)
    return {"available": True, "accepting": public_capacity_available()}


@router.post("/internal/public/reservations")
def reserve_public_capacity(request: Request) -> dict[str, str]:
    """Atomically reserve public and total slots before public work is started."""
    require_loopback(request)
    return {"reservation": reserve()}
