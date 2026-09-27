"""Settings: defaults, blank values, and agreement with .env.example."""

from __future__ import annotations

import re
from pathlib import Path

from harness_config import DYNAMIC_VARIABLES, Settings, settings

ROOT = Path(__file__).resolve().parents[3]
# Read by packages/public_gateway, which keeps its own configuration.
PUBLIC_GATEWAY = re.compile(r"^(PUBLIC_|TURNSTILE_)")


def _aliases() -> set[str]:
    names: set[str] = set()
    for field in Settings.model_fields.values():
        alias = field.validation_alias
        names.update(getattr(alias, "choices", [alias]))
    return {str(name) for name in names}


def test_every_env_example_variable_is_a_setting() -> None:
    variables = re.findall(r"^#?\s*([A-Z][A-Z0-9_]+)=", (ROOT / ".env.example").read_text(), re.MULTILINE)
    unknown = [
        name for name in variables
        if name not in _aliases() and not PUBLIC_GATEWAY.match(name)
        and not name.startswith(DYNAMIC_VARIABLES)
    ]
    assert unknown == []


def test_blank_values_mean_unset_and_flags_parse(monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_AGENT_CONTEXT_TOKENS", "")
    monkeypatch.setenv("HARNESS_MODEL_EDC", " ")
    monkeypatch.setenv("HARNESS_INCLUDE_FIXTURES", "false")
    monkeypatch.setenv("HARNESS_MODEL_STREAM", "yes")
    config = settings()
    assert config.agent_context_tokens is None and config.model_edc is None
    assert config.include_fixtures is False and config.model_stream is True


def test_settings_follow_the_current_environment(monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_MODEL_AGENT", "a")
    assert settings().model_agent == "a"
    monkeypatch.setenv("HARNESS_MODEL_AGENT", "b")
    assert settings().model_agent == "b"


def test_run_deadline_prefers_the_most_specific_variable(monkeypatch) -> None:
    monkeypatch.delenv("HARNESS_RUN_DEADLINE_AGENTIC_SLOW_S", raising=False)
    monkeypatch.delenv("HARNESS_RUN_DEADLINE_SLOW_S", raising=False)
    assert settings().run_deadline_seconds("agentic", "slow") == 300.0
    monkeypatch.setenv("HARNESS_RUN_DEADLINE_SLOW_S", "600")
    assert settings().run_deadline_seconds("baseline", "slow") == 600.0
    monkeypatch.setenv("HARNESS_RUN_DEADLINE_BASELINE_SLOW_S", "120")
    assert settings().run_deadline_seconds("baseline", "slow") == 120.0
    # External: a safety net above the agent's own wall time; the budget-wide variable does not apply.
    monkeypatch.setenv("HARNESS_RUN_DEADLINE_SLOW_S", "900")
    monkeypatch.delenv("HARNESS_RUN_DEADLINE_EXTERNAL_SLOW_S", raising=False)
    assert settings().run_deadline_seconds("external", "slow") == 960.0
    monkeypatch.setenv("HARNESS_RUN_DEADLINE_EXTERNAL_SLOW_S", "1200")
    assert settings().run_deadline_seconds("external", "slow") == 1200.0


def test_model_options_are_read_per_alias(monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_MODEL_OPTIONS", '{"heavy-model": {"max_tokens": 10}}')
    assert settings().model_options("heavy-model") == {"max_tokens": 10}
    assert settings().model_options("lite-model") == {}
    monkeypatch.setenv("HARNESS_MODEL_OPTIONS", "not json")
    assert settings().model_options("heavy-model") == {}


def test_blank_fixture_flag_keeps_the_default_and_bad_values_name_the_variable(monkeypatch) -> None:
    import pytest

    monkeypatch.setenv("HARNESS_INCLUDE_FIXTURES", "")
    assert settings().include_fixtures is True
    monkeypatch.setenv("HARNESS_AGENT_REFLECTIONS", "many")
    with pytest.raises(ValueError, match="HARNESS_AGENT_REFLECTIONS"):
        settings()


def test_settings_are_reused_until_the_environment_changes(monkeypatch) -> None:
    first = settings()
    assert settings() is first
    monkeypatch.setenv("HARNESS_MODEL_JUDGE", "other")
    assert settings() is not first and settings().model_judge == "other"


def test_a_comment_after_an_empty_value_is_not_the_value(tmp_path, monkeypatch) -> None:
    from harness_config import load_env_file, settings

    env = tmp_path / ".env"
    env.write_text("HARNESS_PREDICATE_RELATIONS_FILE=   # empty = the default file\n"
                   "HARNESS_MODEL_RERANK=              # optional\nHARNESS_MODEL_JUDGE=judge-model   # a comment\n")
    for name in ("HARNESS_PREDICATE_RELATIONS_FILE", "HARNESS_MODEL_RERANK", "HARNESS_MODEL_JUDGE"):
        monkeypatch.delenv(name, raising=False)
    load_env_file(env)
    assert settings().predicate_relations_file == ""
    assert settings().model_rerank == ""
    assert settings().model_judge == "judge-model"


def test_env_example_puts_comments_of_empty_values_on_their_own_line() -> None:
    lines = re.findall(r"^[A-Z][A-Z0-9_]*=[ \t]+#.*$", (ROOT / ".env.example").read_text(), re.MULTILINE)
    assert lines == []
