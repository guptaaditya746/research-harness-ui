"""Structured model calls: schema inference, triple extraction, review and answers.

Modules:
- ``model_client``: the OpenAI-compatible JSON client (``OpenAICompatibleModel``) and ``complete``.
- ``call_control``: run deadline, stop requests, cancellation of in-flight calls.
- ``events``: labels and progress events for model calls; the prompt behind each call.
- ``schemas``: Pydantic models for the structured replies.
- ``engine``: ``ExtractionEngine``, the steps shared by all research paths.
- ``claims``: claim subgraphs built from extracted triples (imported by ``labelling``).

Everything below is re-exported so ``from harness_extraction import X`` keeps working.
"""

from __future__ import annotations

import httpx  # noqa: F401 - tests patch harness_extraction.httpx.Client

from .call_control import (
    bounded_timeout,
    call_budget,
    call_interrupted,
    phase_budget,
    run_interruptible,
)
from .engine import ExtractionEngine, local_name
from .errors import ModelFailure, UnsupportedAnswer
from .events import (
    current_prompt,
    current_purpose,
    emit_model_event,
    labelled,
    model_events,
    model_purpose,
    using_prompt,
)
from .model_client import (
    ModelClient,
    OpenAICompatibleModel,
    complete,
    model_options,
    strip_reasoning,
)
from .schemas import (
    AnswerResult,
    EDCDefinition,
    EDCDefinitionResult,
    EDCOpenTriple,
    ExtractedTriple,
    ExtractionResult,
    SchemaInference,
)

__all__ = [
    "AnswerResult",
    "EDCDefinition",
    "EDCDefinitionResult",
    "EDCOpenTriple",
    "ExtractedTriple",
    "ExtractionEngine",
    "ExtractionResult",
    "ModelClient",
    "ModelFailure",
    "OpenAICompatibleModel",
    "SchemaInference",
    "UnsupportedAnswer",
    "bounded_timeout",
    "call_budget",
    "call_interrupted",
    "complete",
    "current_prompt",
    "current_purpose",
    "emit_model_event",
    "labelled",
    "local_name",
    "model_events",
    "model_options",
    "model_purpose",
    "phase_budget",
    "run_interruptible",
    "strip_reasoning",
    "using_prompt",
]
