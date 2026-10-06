"""Repository paths used by the API, with the settings that override them."""

from __future__ import annotations

from pathlib import Path

from harness_config import settings

ROOT = Path(__file__).resolve().parents[1]


def corpus_root() -> Path:
    return settings().corpus_root or ROOT / "corpus_data"


def artifact_root() -> Path:
    return settings().artifact_root or ROOT / "artifacts"


def datasets_root() -> Path:
    return settings().datasets_root or ROOT / "datasets"


def fixtures_dir() -> Path:
    return settings().fixtures_dir or ROOT / "web" / "fixtures"


def evidence_doc_root() -> Path:
    """labelling's evidence documents (<question id>/doc.json), the input of "evidence_doc" runs."""
    return settings().evidence_doc_root or datasets_root() / "evidence_docs"
