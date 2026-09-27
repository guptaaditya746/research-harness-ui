"""Data endpoints: gold candidates, relation registry, corpus papers, question library, budgets."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import app_state
from api_models import CandidateUpdate, RelationReviewRequest, SettingsUpdate
from fastapi import APIRouter, HTTPException
from harness_config import settings
from harness_corpus.store import CorpusStore
from paths import corpus_root, datasets_root
from runtime import external_configured

router = APIRouter()
LOGGER = logging.getLogger(__name__)


@router.get("/datasets/candidates")
def list_candidates() -> list[dict]:
    return app_state.runtime.gold.list()


@router.get("/schema/relations")
def list_relations() -> list[dict]:
    return app_state.runtime.relations.list()


@router.put("/schema/relations/{relation_id}")
def review_relation(relation_id: str, request: RelationReviewRequest) -> dict:
    try:
        return app_state.runtime.relations.review(relation_id, request.status)
    except KeyError as exc:
        raise HTTPException(404, "relation not found") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc



@router.put("/datasets/candidates/{candidate_id}")
def update_candidate(candidate_id: str, request: CandidateUpdate) -> dict:
    changes = request.model_dump(exclude_none=True)
    try:
        return app_state.runtime.gold.update(candidate_id, changes)
    except KeyError as exc:
        raise HTTPException(404, "candidate not found") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc



@router.get("/corpus/papers")
def papers() -> list[dict]:
    store = CorpusStore(corpus_root())
    return store.papers


def _question_sets() -> list[Path]:
    root = datasets_root()
    configured = settings().question_sets
    paths = []
    for name in configured.split(","):
        name = name.strip()
        if name:
            path = Path(name)
            paths.append(path if path.is_absolute() else root / path)
    return paths


@router.get("/questions")
def questions() -> dict[str, list[dict]]:
    """Questions the UI offers for one-click runs: the question sets, plus gold rows and recent runs
    when HARNESS_QUESTION_EXTRAS is on."""
    items: list[dict] = []
    seen: set[str] = set()

    def add(question: object, **meta: object) -> None:
        text = " ".join(str(question or "").split())
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            items.append({"question": text, **meta})

    for path in _question_sets():
        try:
            rows = json.loads(path.read_text())
        except (OSError, ValueError):
            LOGGER.warning("Question set %s is unreadable", path)
            continue
        for row in rows if isinstance(rows, list) else rows.get("questions", []):
            if isinstance(row, dict):
                add(
                    row.get("question"),
                    id=row.get("id"),
                    category=row.get("category", "curated"),
                    type=row.get("type"),
                    theme=row.get("theme"),
                    expected=row.get("expected"),
                    source=path.stem,
                )
    if not settings().question_extras:
        return {"questions": items}
    for row in app_state.runtime.gold.list():
        add(
            row.get("question"),
            id=row.get("id"),
            category="gold",
            type=row.get("status"),
            source="gold",
        )
    for run in app_state.runtime.artifacts.list_runs()[:50]:
        if run.path in {"agentic", "baseline", "baseline-enhanced", "external"} and not run.evaluation_mode:
            add(run.question, id=run.id, category="recent", type=run.status, source="runs")
    recent = [item for item in items if item["category"] == "recent"][:10]
    return {"questions": [item for item in items if item["category"] != "recent"] + recent}


@router.get("/settings")
def get_budget_settings() -> dict[str, str]:
    """The budgets, plus whether the External path is configured (``external``: on / off)."""
    return {**app_state.runtime.settings, "external": "on" if external_configured() else "off"}


@router.put("/settings")
def update_budget_settings(request: SettingsUpdate) -> dict[str, str]:
    changes = request.model_dump(exclude_none=True)
    if any(value not in {"fast", "slow"} for value in changes.values()):
        raise HTTPException(422, "budgets must be fast or slow")
    app_state.runtime.settings.update(changes)
    return dict(app_state.runtime.settings)
