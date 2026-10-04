"""Settings profiles: profiles/<name>.env between the environment and the .env."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from harness_config import load_env_file, profile_file, settings

ROOT = Path(__file__).resolve().parents[3]


def test_profile_values_sit_between_the_environment_and_the_env_file(tmp_path, monkeypatch) -> None:
    (tmp_path / "profiles").mkdir()
    (tmp_path / "profiles" / "wide.env").write_text("HARNESS_SCHEMA_SCOPE=broad\nBASELINE_RETRIEVE_K=60\n"
                                                    "HARNESS_GRAPH_LINKING=dense\n")
    env = tmp_path / ".env"
    env.write_text("HARNESS_PROFILE=wide\nHARNESS_SCHEMA_SCOPE=focused\nHARNESS_EDC_BATCH_PASSAGES=3\n")
    for name in ("HARNESS_PROFILE", "HARNESS_SCHEMA_SCOPE", "BASELINE_RETRIEVE_K", "HARNESS_EDC_BATCH_PASSAGES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HARNESS_GRAPH_LINKING", "standard")       # the environment wins
    load_env_file(env)
    try:
        config = settings()
        assert config.schema_scope == "broad"                     # profile over .env
        assert config.baseline_retrieve_k == 60
        assert config.edc_batch_passages == 3                     # .env when the profile is silent
        assert config.graph_linking == "standard"
        assert config.profile == "wide"
    finally:
        for name in ("HARNESS_PROFILE", "HARNESS_SCHEMA_SCOPE", "BASELINE_RETRIEVE_K", "HARNESS_EDC_BATCH_PASSAGES"):
            os.environ.pop(name, None)


def test_a_missing_profile_is_an_error(tmp_path, monkeypatch) -> None:
    env = tmp_path / ".env"
    env.write_text("HARNESS_PROFILE=nope\n")
    monkeypatch.delenv("HARNESS_PROFILE", raising=False)
    with pytest.raises(ValueError, match="no profile file"):
        load_env_file(env)


def test_the_shipped_exhaustive_profile_only_sets_known_variables(monkeypatch) -> None:
    from dotenv import dotenv_values

    values = dotenv_values(profile_file("exhaustive", ROOT))
    assert values["HARNESS_UNMAPPED_RELATIONS"] == "keep" and values["HARNESS_GRAPH_LINKING"] == "dense"
    for key, value in values.items():
        monkeypatch.setenv(key, value or "")
    settings()                                                    # every value parses
