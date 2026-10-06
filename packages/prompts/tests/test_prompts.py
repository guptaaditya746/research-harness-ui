"""Prompt registry: every v2 prompt renders, payload order, examples and hashes."""

from __future__ import annotations

import json

import pytest
from harness_prompts import SETS, names, render, text

SUPERVISOR_QUESTION = "Which benchmarks recur across papers, and how many papers cite them?"


def _variables(name: str) -> dict:
    from harness_agent.skills import catalog, relevant

    loaded = relevant(SUPERVISOR_QUESTION)
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
        "agent/evidence_doc_context": {"context_json": ""},
        "planning/query_plan": {"operations": {}},
    }.get(name, {})


def test_only_v2_exists_and_it_defines_every_prompt() -> None:
    assert SETS == ("v2",)
    assert len(names()) == 21


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


def test_unknown_set_and_prompt_are_errors(monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_PROMPT_SET", "v9")
    with pytest.raises(ValueError, match="HARNESS_PROMPT_SET"):
        render("extraction/edc_define")
    with pytest.raises(KeyError):
        render("extraction/nope", prompt_set="v2")


def test_version_is_the_template_hash_and_a_deleted_set_is_refused() -> None:
    a = render("extraction/edc_define")
    assert a.ref == f"extraction/edc_define@v2:{a.version}" and len(a.version) == 8
    assert "\n- x\n" in text("agent/decision_rejected", problems=["x"])
    with pytest.raises(ValueError, match="unknown prompt set"):
        render("extraction/edc_define", prompt_set="v1")


EXAMPLE_PROMPTS = ["extraction/edc_extract", "extraction/evidence_review", "extraction/answer_composition",
                   "agent/paper_reader", "agent/research_supervisor"]


@pytest.mark.parametrize("name", EXAMPLE_PROMPTS)
def test_examples_are_shown_and_change_the_hash(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    import harness_prompts

    plain = render(name, prompt_set="v2", **_variables(name))
    assert "Examples from earlier errors" not in plain.system
    examples = [{"input": "passage p1-2: \"we evaluate on CALVIN\"", "output": "{\"triples\": []}",
                 "why": "the benchmark was missed"},
                {"input": "an example without a reason", "output": "{}"}]
    raw = json.dumps(examples)
    monkeypatch.setattr(harness_prompts, "_examples",
                        lambda prompt_set, prompt: (raw, examples) if prompt == name else ("", []))
    shown = render(name, prompt_set="v2", **_variables(name))
    assert "Examples from earlier errors" in shown.system
    assert "Why: the benchmark was missed" in shown.system
    assert shown.version != plain.version


def test_shipped_example_files_are_valid() -> None:
    from importlib import resources

    root = resources.files("harness_prompts").joinpath("examples")
    for prompt_set in SETS:
        folder = root.joinpath(prompt_set)
        if not folder.is_dir():
            continue
        stack = list(folder.iterdir())
        while stack:
            entry = stack.pop()
            if entry.is_dir():
                stack.extend(entry.iterdir())
                continue
            if not entry.name.endswith(".json"):
                continue
            relative = str(entry).split(f"examples/{prompt_set}/", 1)[1].removesuffix(".json")
            assert relative in EXAMPLE_PROMPTS, f"{relative}: no template shows examples"
            examples = json.loads(entry.read_text())
            assert isinstance(examples, list)
            assert all({"input", "output"} <= set(item) for item in examples), relative
            assert not any("TODO" in str(item.get("why", "")) for item in examples), f"{relative}: unfinished why"
