from harness_agent.context import (
    Conversation,
    cap_tool_result,
    estimate_tokens,
    is_context_error,
    window_from_error,
)


def exchange(index: int, size: int = 3000) -> tuple[dict, list[dict]]:
    assistant = {"role": "assistant", "content": None, "tool_calls": [
        {"id": f"c{index}a", "type": "function", "function": {"name": "search", "arguments": "{}"}},
        {"id": f"c{index}b", "type": "function", "function": {"name": "search", "arguments": "{}"}},
    ]}
    results = [{"role": "tool", "tool_call_id": f"c{index}{s}", "content": "x" * size} for s in "ab"]
    return assistant, results


def assert_valid(messages: list[dict]) -> None:
    """Every tool message answers a call requested by the preceding assistant turn."""
    pending: set[str] = set()
    for message in messages:
        if message["role"] == "assistant" and message.get("tool_calls"):
            assert not pending, "previous tool calls were left unanswered"
            pending = {call["id"] for call in message["tool_calls"]}
        elif message["role"] == "tool":
            assert message["tool_call_id"] in pending, "orphaned tool message"
            pending.discard(message["tool_call_id"])
    assert not pending


def test_fitting_drops_whole_exchanges_and_keeps_a_valid_conversation() -> None:
    conversation = Conversation([{"role": "system", "content": "s"}, {"role": "user", "content": "q"},
                                 {"role": "user", "content": "grounded"}])
    for index in range(6):
        conversation.add_exchange(*exchange(index))
    assert conversation.fit(4000, [], '{"evidence_ids": ["p4-results"]}')
    messages = conversation.messages()
    assert conversation.tokens([]) <= 4000
    assert [m["content"] for m in messages[:3]] == ["s", "q", "grounded"]
    assert "p4-results" in messages[3]["content"]
    assert_valid(messages)


def test_single_oversized_exchange_is_truncated_not_broken() -> None:
    conversation = Conversation([{"role": "system", "content": "s"}])
    conversation.add_exchange(*exchange(0, size=60_000))
    conversation.fit(5000, [], "{}")
    assert conversation.tokens([]) <= 5200
    assert_valid(conversation.messages())
    assert "truncated" in conversation.messages()[-1]["content"]


def test_provider_length_errors_are_recognised() -> None:
    vllm = ("This model's maximum context length is 16384 tokens. However, you requested "
            "21000 tokens (18952 in the messages, 2048 in the completion).")
    assert is_context_error(vllm)
    assert window_from_error(vllm) == 16384
    assert not is_context_error("Invalid tool schema")


def test_tool_results_are_bounded() -> None:
    capped = cap_tool_result("y" * 100_000, limit_tokens=4000)
    assert len(capped) < 4000 and "truncated" in capped
    assert estimate_tokens("abc" * 100) == 101


def test_grounded_context_shrinks_in_answer_order_with_an_explicit_note() -> None:
    claims = [{"id": f"claim-{i}", "fields": {"object": "x" * 200}} for i in range(80)]
    conversation = Conversation([{"role": "system", "content": "s"}],
                                grounded={"preliminary_answer": "a", "preliminary_claims": claims})
    assert conversation.fit(3000, [], "{}")
    grounded = conversation.messages()[1]["content"]
    assert conversation.tokens([]) <= 3000
    assert '"claim-0"' in grounded and '"claim-79"' not in grounded
    assert "omitted_for_context" in grounded and "structured_facts" in grounded


def test_older_tool_results_are_masked_by_digests_and_stay_valid() -> None:
    conversation = Conversation([{"role": "system", "content": "s"}], keep_results=2)
    for index in range(4):
        assistant, results = exchange(index, size=2000)
        conversation.add_exchange(assistant, results, [f'{{"digest": {index}}}'] * 2)
    messages = conversation.messages()
    assert_valid(messages)
    tool_contents = [m["content"] for m in messages if m["role"] == "tool"]
    assert tool_contents[:4] == ['{"digest": 0}'] * 2 + ['{"digest": 1}'] * 2
    assert tool_contents[4:] == ["x" * 2000] * 4
    assert conversation.masked_results == 4
    assert conversation.exchanges[0][1]["content"] == "x" * 2000       # the originals are kept


def test_calibration_moves_the_estimate_towards_the_server_count() -> None:
    conversation = Conversation([{"role": "user", "content": "x" * 3000}])
    raw = conversation.raw_tokens([])
    conversation.calibrate(raw * 2, raw)
    assert conversation.scale == 1.5 and conversation.tokens([]) == int(raw * 1.5)
    conversation.calibrate(None, raw)
    conversation.calibrate(10 * raw, raw)                               # bounded at 2.5
    assert conversation.scale == 2.0


def test_archiving_replaces_older_exchanges_with_a_summary_and_the_ledger() -> None:
    conversation = Conversation([{"role": "system", "content": "s"}])
    for index in range(3):
        conversation.add_exchange(*exchange(index, size=100), ["d", "d"])
    conversation.archive_oldest(2, "searched SafeRAG; found p1-a", '{"your_findings": []}')
    messages = conversation.messages()
    assert_valid(messages)
    assert len(conversation.exchanges) == 1 and conversation.dropped_exchanges == 2
    assert "Summary of the archived research: searched SafeRAG; found p1-a" in messages[1]["content"]
