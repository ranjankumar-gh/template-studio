from __future__ import annotations

from templatestudio.tools import (
    HermesParser,
    JSONBlockParser,
    tool_call_message,
    tool_result_message,
)


def test_tool_call_message_shape():
    msg = tool_call_message("get_order", {"id": "4471"})
    assert msg["role"] == "assistant"
    assert isinstance(msg["tool_calls"], list)
    call = msg["tool_calls"][0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "get_order"
    assert call["function"]["arguments"] == {"id": "4471"}


def test_tool_result_message_shape():
    msg = tool_result_message("get_order", "shipped")
    assert msg == {"role": "tool", "name": "get_order", "content": "shipped"}


def test_json_block_parser_extracts_call():
    text = 'sure:\n```json\n{"name": "get_order", "arguments": {"id": "4471"}}\n```'
    calls = JSONBlockParser().parse(text)
    assert len(calls) == 1
    assert calls[0].name == "get_order"
    assert calls[0].arguments == {"id": "4471"}


def test_json_block_parser_normalizes_stringified_arguments():
    text = '```json\n{"name": "get_order", "arguments": "{\\"id\\": \\"4471\\"}"}\n```'
    calls = JSONBlockParser().parse(text)
    assert len(calls) == 1
    assert calls[0].arguments == {"id": "4471"}


def test_json_block_parser_skips_invalid_json():
    text = "```json\nnot json\n```"
    assert JSONBlockParser().parse(text) == []


def test_json_block_parser_handles_no_block():
    assert JSONBlockParser().parse("just text") == []


def test_json_block_parser_skips_json_that_names_no_tool():
    """A model that answers with a config example must not crash the loop."""
    text = 'Use this:\n```json\n{"timeout": 30}\n```'
    assert JSONBlockParser().parse(text) == []
    assert JSONBlockParser().parse("```json\n[1, 2]\n```") == []


def test_hermes_parser_extracts_qwen3_tool_calls():
    """Qwen3's template writes calls as <tool_call>, a JSON object, and
    </tool_call>, one block per call."""
    text = (
        "Checking both.\n"
        '<tool_call>\n{"name": "get_order", "arguments": {"id": "4471"}}\n'
        "</tool_call>\n"
        '<tool_call>\n{"name": "get_temperature", "arguments": '
        '{"location": "Paris"}}\n</tool_call>'
    )
    calls = HermesParser().parse(text)
    assert [c.name for c in calls] == ["get_order", "get_temperature"]
    assert calls[1].arguments == {"location": "Paris"}


def test_each_parser_ignores_the_other_format():
    """The opening bug: the right call, in a format the parser does not read."""
    fenced = '```json\n{"name": "get_order", "arguments": {}}\n```'
    tagged = '<tool_call>\n{"name": "get_order", "arguments": {}}\n</tool_call>'
    assert HermesParser().parse(fenced) == []
    assert JSONBlockParser().parse(tagged) == []


# --- the agent loop ---

from templatestudio.tools import (  # noqa: E402
    mcp_result_to_tool_message,
    parallel_tool_call_message,
    validate_tool_pairing,
)


def _loop():
    a = tool_call_message("get_order", {"id": "4471"}, call_id="call_a")
    b = tool_call_message("get_temperature", {"location": "Paris"}, call_id="call_b")
    return [
        {"role": "user", "content": "Where is #4471 and is it warm there?"},
        parallel_tool_call_message(a, b),
        tool_result_message("get_order", "shipped", tool_call_id="call_a"),
        tool_result_message("get_temperature", "22", tool_call_id="call_b"),
    ]


def test_parallel_message_carries_both_calls_with_ids():
    msg = _loop()[1]
    assert [c["id"] for c in msg["tool_calls"]] == ["call_a", "call_b"]


def test_paired_loop_validates():
    assert validate_tool_pairing(_loop()) == []


def test_results_may_arrive_out_of_order_when_ids_pair_them():
    loop = _loop()
    loop[2], loop[3] = loop[3], loop[2]
    assert validate_tool_pairing(loop) == []


def test_truncation_that_drops_the_call_orphans_the_result():
    loop = _loop()
    del loop[1]
    assert "answers no call" in validate_tool_pairing(loop)[0]


def test_a_result_for_an_unknown_id_is_flagged():
    loop = _loop()
    loop[3] = tool_result_message("get_temperature", "22", tool_call_id="call_z")
    assert "matches no call" in validate_tool_pairing(loop)[0]


def test_a_user_turn_before_every_result_is_flagged():
    loop = _loop()[:3] + [{"role": "user", "content": "hello?"}]
    assert "1 call(s) unanswered" in validate_tool_pairing(loop)[0]


def test_an_assistant_turn_before_every_result_is_flagged():
    """Truncation kept call_a's result and dropped call_b's; the model replied."""
    loop = _loop()[:3] + [{"role": "assistant", "content": "It shipped."}]
    assert validate_tool_pairing(loop) == ["message 3: 1 call(s) unanswered"]


def test_history_that_ends_with_an_unanswered_call_is_flagged():
    loop = _loop()[:3]
    assert validate_tool_pairing(loop) == [
        "end of conversation: 1 call(s) unanswered"
    ]


def test_mcp_pretty_printed_text_block_is_not_duplicated():
    """Servers send structuredContent as a text block too, often indented."""
    result = {
        "content": [{"type": "text", "text": '{\n  "a": 1\n}'}],
        "structuredContent": {"a": 1},
    }
    msg = mcp_result_to_tool_message("f", result)
    assert msg["content"] == '{\n  "a": 1\n}'


def test_mcp_result_keeps_text_and_serializes_structured_content():
    result = {
        "content": [{"type": "text", "text": "shipped"},
                    {"type": "image", "mimeType": "image/png", "data": "..."}],
        "structuredContent": {"status": "shipped", "eta_days": 2},
    }
    msg = mcp_result_to_tool_message("get_order", result, tool_call_id="call_a")
    assert msg["content"] == (
        'shipped\n[image omitted: image/png]\n{"status": "shipped", "eta_days": 2}'
    )
    assert msg["tool_call_id"] == "call_a"


def test_mcp_structured_only_result_does_not_render_empty():
    result = {"content": [], "structuredContent": {"a": 1}}
    msg = mcp_result_to_tool_message("f", result)
    assert msg["content"] == '{"a": 1}'


def test_mcp_error_is_marked():
    msg = mcp_result_to_tool_message(
        "f", {"content": [{"type": "text", "text": "boom"}], "isError": True})
    assert msg["content"] == "[tool error]\nboom"
