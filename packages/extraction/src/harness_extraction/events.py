"""Model-call observability: what each call is for, and where its events go.

Callers label calls (``model_purpose`` / ``labelled``); the runtime installs
an event callback per run (``model_events``). The prompt behind a call is
recorded with ``using_prompt`` so traces name the exact template version.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any

_MODEL_PURPOSE: ContextVar[str] = ContextVar("harness_model_purpose", default="model call")
_MODEL_EVENTS: ContextVar[Callable[[str, str], None] | None] = ContextVar(
    "harness_model_events", default=None
)
_PROMPT_REF: ContextVar[str | None] = ContextVar("harness_prompt_ref", default=None)


@contextmanager
def model_purpose(purpose: str) -> Iterator[None]:
    """Label the model calls made inside the block (e.g. "schema inference")."""
    token = _MODEL_PURPOSE.set(purpose)
    try:
        yield
    finally:
        _MODEL_PURPOSE.reset(token)


@contextmanager
def model_events(callback: Callable[[str, str], None]) -> Iterator[None]:
    """Receive (action, detail) events for every model call inside the block."""
    token = _MODEL_EVENTS.set(callback)
    try:
        yield
    finally:
        _MODEL_EVENTS.reset(token)


@contextmanager
def using_prompt(ref: str | None) -> Iterator[None]:
    """Record which prompt template (``name@set:version``) the calls inside use."""
    token = _PROMPT_REF.set(ref)
    try:
        yield
    finally:
        _PROMPT_REF.reset(token)


def labelled(purpose: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: label every model call made by the wrapped function."""
    def wrap(function: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(function)
        def inner(*args: Any, **kwargs: Any) -> Any:
            with model_purpose(purpose):
                return function(*args, **kwargs)
        return inner
    return wrap


def emit_model_event(action: str, detail: str) -> None:
    callback = _MODEL_EVENTS.get()
    if callback is not None:
        callback(action, detail)


def current_purpose() -> str:
    return _MODEL_PURPOSE.get()


def current_prompt() -> str | None:
    return _PROMPT_REF.get()


__all__ = [
    "current_prompt", "current_purpose", "emit_model_event", "labelled", "model_events",
    "model_purpose", "using_prompt",
]
