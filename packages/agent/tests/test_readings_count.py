"""How many concept readings: READINGS_MIN..READINGS_MAX used as given (one for an unambiguous question)."""

from __future__ import annotations

import pytest
from harness_agent.concepts import ConceptProposalError, propose_grounded
from harness_config import settings


class Model:
    model = "stub"

    def __init__(self, count: int) -> None:
        self.count = count
        self.systems: list[str] = []

    def complete_json(self, system: str, _user: str) -> dict:
        self.systems.append(system)
        return {"concepts": [{"label": f"Reading {i}", "gloss": f"Research MethodX by measure {i}.",
                              "classes": ["Paper"], "search_terms": [f"term {i}"]} for i in range(self.count)]}


def test_defaults_allow_a_single_reading_and_the_prompt_asks_for_one_when_unambiguous(monkeypatch) -> None:
    monkeypatch.delenv("READINGS_MIN", raising=False)
    monkeypatch.delenv("READINGS_MAX", raising=False)
    assert (settings().readings_min, settings().readings_max) == (1, 4)
    model = Model(1)
    proposal = propose_grounded(model, "Which benchmark does MethodX report on?", {"Paper"})
    assert [item.label for item in proposal.concepts] == ["Reading 0"]
    assert len(model.systems) == 1                            # one reading is enough: no retry
    assert "propose a single reading" in model.systems[0]
    assert "genuinely ambiguous" in model.systems[0]


@pytest.mark.parametrize(("low", "high", "given", "kept"), [("1", "1", 1, 1), ("2", "6", 6, 6), ("1", "2", 3, 2)])
def test_bounds_are_used_as_configured(monkeypatch, low, high, given, kept) -> None:
    monkeypatch.setenv("READINGS_MIN", low)
    monkeypatch.setenv("READINGS_MAX", high)
    assert len(propose_grounded(Model(given), "Q about BenchY?", {"Paper"}).concepts) == kept


def test_a_minimum_above_the_maximum_is_rejected_not_clamped(monkeypatch) -> None:
    monkeypatch.setenv("READINGS_MIN", "3")
    monkeypatch.setenv("READINGS_MAX", "2")
    model = Model(3)
    with pytest.raises(ConceptProposalError, match="READINGS_MIN"):
        propose_grounded(model, "Q about BenchY?", {"Paper"})
    assert model.systems == []                                # rejected before any model call


def test_a_minimum_below_one_is_an_invalid_setting(monkeypatch) -> None:
    monkeypatch.setenv("READINGS_MIN", "0")
    with pytest.raises(ValueError, match="READINGS_MIN"):
        settings()
