"""Prompt registry: v1 reproduces the pre-template prompts exactly; v2 is complete and renders."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from harness_prompts import SETS, names, render, text

GOLDEN = Path(__file__).parent / "golden"
INPUTS = json.loads((GOLDEN / "_inputs.json").read_text())


def _variables(name: str) -> dict:
    from harness_agent.skills import catalog, relevant

    loaded = relevant(INPUTS["supervisor_question"])
    return {
        "agent/concept_proposal": {"minimum": 3, "maximum": 4},
        "agent/research_supervisor": {
            "skills_catalog": catalog(),
            "loaded_skills": "; ".join(f"{s.name}@{s.version}: {s.instructions}" for s in loaded),
            "skills": list(loaded),
        },
        "agent/decision_rejected": {"problems": ["decision cites unobserved ids: p99",
                                                 "no corpus tool was called"]},
        "agent/grounded_context": {"context_json": ""},
        "planning/query_plan": {"operations": {}},
    }.get(name, {})


@pytest.mark.parametrize("name", names("v1"))
def test_v1_matches_the_prompt_sent_before_templating(name: str) -> None:
    golden_name = name.replace("/", "__").replace("grounded_context", "grounded_context_prefix")
    expected = (GOLDEN / f"{golden_name}.txt").read_text()
    assert render(name, prompt_set="v1", **_variables(name)).system == expected


def test_both_sets_define_the_same_prompts() -> None:
    assert set(names("v1")) == set(names("v2"))
    assert len(names("v1")) == 16


@pytest.mark.parametrize("name", names("v2"))
def test_v2_renders_without_missing_variables(name: str) -> None:
    from harness_baseline.planner import OPERATIONS

    variables = _variables(name)
    if name == "planning/query_plan":
        variables = {"operations": OPERATIONS}
    prompt = render(name, {"question": "q", "passages": [{"passage_id": "p1", "text": "t"}]},
                    prompt_set="v2", **variables)
    assert prompt.system.strip()
    assert prompt.ref.startswith(f"{name}@v2:")


def test_v2_payload_puts_documents_first_and_question_last() -> None:
    prompt = render("extraction/triple_extraction",
                    {"question": "q", "classes": ["Paper"], "passages": [{"passage_id": "p1"}]},
                    prompt_set="v2")
    assert list(json.loads(prompt.user)) == ["passages", "classes", "question"]
    legacy = render("extraction/triple_extraction",
                    {"question": "q", "classes": ["Paper"], "passages": [{"passage_id": "p1"}]},
                    prompt_set="v1")
    assert list(json.loads(legacy.user)) == ["question", "classes", "passages"]


def test_unknown_set_and_prompt_are_errors(monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_PROMPT_SET", "v9")
    with pytest.raises(ValueError, match="unknown prompt set"):
        render("extraction/edc_define")
    with pytest.raises(KeyError):
        render("extraction/nope", prompt_set="v2")


def test_version_changes_with_the_template_and_set() -> None:
    a = render("extraction/edc_define", prompt_set="v1")
    b = render("extraction/edc_define", prompt_set="v2")
    assert a.version != b.version and set(SETS) == {"v1", "v2"}
    assert text("agent/decision_rejected", prompt_set="v1", problems=["x"]).endswith("\n- x")
