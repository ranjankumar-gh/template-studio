from __future__ import annotations

from templatestudio.reasoning import (
    assistant_message,
    has_empty_think_blocks,
    strip_reasoning,
)


def test_strip_reasoning_removes_think_block():
    generated = "<think>\nlet me think\n</think>\nParis."
    assert strip_reasoning(generated) == "Paris."


def test_strip_reasoning_leaves_plain_text():
    assert strip_reasoning("Paris.") == "Paris."


def test_has_empty_think_blocks_detects_empty():
    assert has_empty_think_blocks("<|im_start|>assistant\n<think></think>\nhi") is True


def test_has_empty_think_blocks_false_on_nonempty():
    assert has_empty_think_blocks("<think>reasoning</think>") is False


def test_assistant_message_carries_reasoning_in_dedicated_field():
    msg = assistant_message("Paris.", reasoning="capital of France")
    assert msg["role"] == "assistant"
    assert msg["content"] == "Paris."
    assert msg["reasoning_content"] == "capital of France"


def test_assistant_message_omits_reasoning_when_none():
    msg = assistant_message("Paris.")
    assert "reasoning_content" not in msg
