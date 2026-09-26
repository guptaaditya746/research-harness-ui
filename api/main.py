"""Research Harness HTTP API: the FastAPI app and its routers.

Where things are:
- ``runtime``: ``RunRuntime``, the run lifecycle (one thread per run).
- ``app_state``: the process-wide runtime instance used by the routes.
- ``capacity``: model slots and public-demo reservations.
- ``api_models``: request bodies.
- ``routes_runs``, ``routes_benchmark``, ``routes_data``, ``routes_public``: endpoints.
- ``paths``: repository paths and their settings overrides.

Run with ``uvicorn main:app --app-dir api`` (see ``make dev`` / ``make demo``).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import routes_benchmark
import routes_data
import routes_public
import routes_runs
import routes_status
from api_models import RunRequest
from app_state import runtime
from capacity import MODEL_SLOT, PUBLIC_MODEL_SLOT
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from harness_config import settings
from harness_observability import hello_span
from harness_observability import init as init_observability
from routes_data import _question_sets
from runtime import RunRuntime, fixture_runs, run_deadline_seconds


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # Telemetry must not delay API availability when the remote collector is
    # slow or temporarily unavailable, including during client initialization.
    asyncio.create_task(asyncio.to_thread(lambda: init_observability() and hello_span()))
    yield


app = FastAPI(title="Research Harness", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings().api_cors_origins.split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)
for module in (routes_public, routes_runs, routes_benchmark, routes_data, routes_status):
    app.include_router(module.router)

__all__ = [
    "MODEL_SLOT", "PUBLIC_MODEL_SLOT", "RunRequest", "RunRuntime", "_question_sets", "app",
    "fixture_runs", "run_deadline_seconds", "runtime",
]
