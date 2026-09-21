from harness_ledger import Ledger, Obligation
from hypothesis import given
from hypothesis import strategies as st


def obligation(
    index: int, *, kind: str = "content", expected_object: str | None = None
) -> Obligation:
    return Obligation(
        id=f"o-{index}",
        subject=f"https://harness.local/resource/p{index}",
        property="https://harness.local/schema#cites",
        kind=kind,
        expected_object=expected_object,
    )


@given(st.sampled_from(["satisfied", "unsupported", "out_of_budget"]))
def test_only_open_obligations_can_transition_once(target: str) -> None:
    ledger = Ledger([obligation(1)])
    if target == "satisfied":
        ledger.satisfy("o-1", ["p1-ref-1"])
    elif target == "unsupported":
        ledger.mark_unsupported(
            "o-1",
            [
                {"tool": "search", "query": "cites", "scope": {"paper_ids": ["p1"]}},
                {"tool": "grep", "query": "cites", "scope": {"paper_ids": ["p1"]}},
            ],
        )
    else:
        ledger.mark_out_of_budget()

    assert ledger.obligations[0].status == target
    try:
        ledger.satisfy("o-1", ["another-source"])
    except ValueError as error:
        assert "not open" in str(error)
    else:
        raise AssertionError("a terminal obligation transitioned twice")


def test_unsupported_content_requires_search_and_grep_records() -> None:
    ledger = Ledger([obligation(1)])
    for searched in ([], [{"tool": "search", "query": "x"}]):
        try:
            ledger.mark_unsupported("o-1", searched)
        except ValueError as error:
            assert "search and grep" in str(error)
        else:
            raise AssertionError("unsupported accepted incomplete search evidence")


def test_structural_obligations_close_from_fact_table() -> None:
    ledger = Ledger(
        [
            obligation(1, kind="structural", expected_object="p17"),
            obligation(2, kind="structural", expected_object="p17"),
        ]
    )
    ledger.close_structural(
        [
            {
                "subject": "p1",
                "predicate": "cites",
                "object": "p17",
                "source_id": "p1-ref-3",
            }
        ]
    )
    assert ledger.closed
    assert ledger.obligations[0].status == "satisfied"
    assert ledger.obligations[0].evidence == ["p1-ref-3"]
    assert ledger.obligations[1].status == "unsupported"
    assert ledger.coverage() == 1.0


@given(st.lists(st.booleans(), min_size=1, max_size=30))
def test_closed_ledger_is_queryable(states: list[bool]) -> None:
    ledger = Ledger([obligation(i) for i in range(len(states))])
    for item, supported in zip(ledger.obligations, states, strict=True):
        if supported:
            ledger.satisfy(item.id, [f"source-{item.id}"])
        else:
            ledger.mark_out_of_budget([item.id])
    assert ledger.closed
    assert ledger.queryable
    assert ledger.coverage() == sum(states) / len(states)


def test_next_is_stable_and_bounded() -> None:
    ledger = Ledger([obligation(i) for i in range(4)])
    assert [item.id for item in ledger.next(2)] == ["o-0", "o-1"]
    assert ledger.next(0) == []
