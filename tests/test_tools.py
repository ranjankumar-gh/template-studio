from __future__ import annotations

from templatestudio.tools import (
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
