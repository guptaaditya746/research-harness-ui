"""Baseline research paths (Fast and Adaptive) as a fixed LangGraph workflow.

Modules:
- ``pipeline``: ``Baseline20``, the run entry point and graph wiring.
- ``retrieval``, ``research_steps``, ``facts``, ``outcomes``: the pipeline's steps (mixins).
- ``state``: pipeline state, results and early outcomes (``NoResults`` and friends).
- ``planner``: question understanding (``QueryPlanner``) and paper resolution.
- ``answer_graph``: the connected answer graph shown in the UI.
"""

from __future__ import annotations

from .pipeline import Baseline20
from .state import (
    BaselineResult,
    BaselineState,
    CorpusAbsent,
    NoEvidenceFound,
    NoResults,
    retrieval_dsl,
)

__all__ = [
    "Baseline20", "BaselineResult", "BaselineState", "CorpusAbsent", "NoEvidenceFound", "NoResults",
    "retrieval_dsl",
]
