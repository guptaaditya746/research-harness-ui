"""Run-level control over model calls: deadline, stop requests, cancellation.

A run sets its wall-clock deadline and stop flag once (``call_budget``); every
model call made on its behalf sees them through a context variable, which
follows the run's thread and its asyncio tasks.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from typing import Any

from harness_config import settings

from .errors import ModelFailure

_CALL_BUDGET: ContextVar[tuple[float, Callable[[], bool]] | None] = ContextVar(
    "harness_call_budget", default=None
)


@contextmanager
def call_budget(deadline: float, stopped: Callable[[], bool]) -> Iterator[None]:
    """Bound every model call inside the block by ``deadline`` (time.monotonic)."""
    token = _CALL_BUDGET.set((deadline, stopped))
    try:
        yield
    finally:
        _CALL_BUDGET.reset(token)


@contextmanager
def phase_budget(seconds: float) -> Iterator[None]:
    """Inside the block, model calls also end ``seconds`` from now (a phase deadline within the run's).

    Blocking calls made through ``run_interruptible`` (also from threads that copy the context, such
    as ``asyncio.to_thread``) are abandoned at that deadline instead of running to their own timeout.
    """
    budget = _CALL_BUDGET.get()
    deadline = time.monotonic() + seconds
    stopped: Callable[[], bool] = lambda: False
    if budget is not None:
        deadline, stopped = min(deadline, budget[0]), budget[1]
    with call_budget(deadline, stopped):
        yield


def bounded_timeout(timeout_s: float) -> float:
    """The per-call timeout, shortened to the time the run has left.

    Raises ModelFailure when the run was stopped or its deadline has passed,
    so no new model call starts after cancellation.
    """
    budget = _CALL_BUDGET.get()
    if budget is None:
        return timeout_s
    deadline, stopped = budget
    if stopped():
        raise ModelFailure("run stopped")
    remaining = deadline - time.monotonic()
    if remaining <= 1.0:
        raise ModelFailure("run deadline reached")
    return min(timeout_s, remaining)


def call_interrupted() -> str | None:
    """Why the current run's model calls must stop, or None."""
    budget = _CALL_BUDGET.get()
    if budget is None:
        return None
    deadline, stopped = budget
    if stopped():
        return "run stopped"
    if time.monotonic() > deadline:
        return "run deadline reached"
    return None


_CALL_POOL = ThreadPoolExecutor(
    max_workers=settings().model_call_threads, thread_name_prefix="model-call"
)


def run_interruptible(call: Callable[[], Any], cancel: threading.Event | None = None) -> Any:
    """Run a blocking provider call that Stop and the run deadline can abandon.

    The HTTP request runs on a worker thread; the run's own thread waits in
    short intervals and gives up as soon as the run is stopped or out of time,
    so the run (and its model slot) is released without waiting for the
    provider. ``cancel`` is set when that happens: a streaming request checks
    it between chunks and closes its connection, which makes the server abort
    the generation. A non-streaming request cannot be interrupted and
    finishes in the background (see docs/KNOWN_ISSUES.md).
    """
    if _CALL_BUDGET.get() is None:
        return call()
    future = _CALL_POOL.submit(copy_context().run, call)
    while True:
        try:
            return future.result(timeout=0.25)
        except FutureTimeout:
            reason = call_interrupted()
            if reason:
                if cancel is not None:
                    cancel.set()
                future.cancel()
                raise ModelFailure(reason) from None


def parallel_map(function: Callable[[Any], Any], items: list[Any], workers: int) -> list[Any]:
    """``[function(item) for item in items]`` with up to ``workers`` items at a time, results in input order.

    Each item runs in a copy of the caller's context, so the run's deadline, stop flag and model-call
    labels apply inside the workers. The first failure (in input order) is raised after the other
    started items finish; items not yet started are cancelled. ``workers <= 1`` runs them one by one.
    """
    if workers <= 1 or len(items) <= 1:
        return [function(item) for item in items]
    pool = ThreadPoolExecutor(max_workers=min(workers, len(items)), thread_name_prefix="harness-batch")
    try:
        futures = [pool.submit(copy_context().run, function, item) for item in items]
        return [future.result() for future in futures]
    except BaseException:
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    finally:
        pool.shutdown(wait=True)


__all__ = ["bounded_timeout", "call_budget", "call_interrupted", "parallel_map", "phase_budget", "run_interruptible"]
