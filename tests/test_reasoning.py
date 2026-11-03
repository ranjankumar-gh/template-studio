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


def test_strip_reasoning_removes_all_blocks():
    generated = "<think>first</think>\nstep one.\n<think>second</think>\ndone."
    assert strip_reasoning(generated) == "step one.\ndone."


def test_strip_reasoning_handles_think_opened_in_the_prompt():
    # The template put <think> in the generation prompt; output has only </think>.
    generated = "let me think\n</think>\n\nParis."
    assert strip_reasoning(generated) == "Paris."


def test_strip_reasoning_never_shows_a_truncated_prompt_opened_block():
    """max_tokens hit before </think>: the output is all reasoning, and it
    must not reach the user as the answer."""
    truncated = "The user's system prompt says never to apologize, so"
    assert strip_reasoning(truncated, prompt_opened=True) == ""
    assert strip_reasoning("ok\n</think>\n\nParis.", prompt_opened=True) == "Paris."


def test_strip_reasoning_drops_an_unclosed_block_the_model_opened():
    assert strip_reasoning("<think>still going and then cut") == ""
    assert strip_reasoning("Paris.\n<think>an afterthought, cut") == "Paris.\n"


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
